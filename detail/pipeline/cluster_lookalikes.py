"""Find look-alike clusters — an analysis module (§16) over a clustering run's tree (§22
"Look-alike clusters").

One executor job compares pairs of one temporal-clustering run's branch tops — every pair,
or (``pairs: labelled``) each labelled branch top against every other — each with
a cluster profile of one against the other (``Experiment.profile_cluster`` with ``compare:
node``, position and facing left out), written into **this job's own run directory**
(``analysis/<an_NN>/profiles/node_<a>/against_<b>/``) and then **placed** beside the clustering
run (``api/profile_describe.place``) exactly where a pair compared by hand in the Cluster profile
window is saved. The groups are read from those placed comparisons
(``api/cluster_lookalikes.py``), so the Interpret tab shows them as soon as a pair is placed,
and a comparison made by hand counts the same. ``analysis.yaml`` is the job's record: which run,
the pairs as resolved at launch (Tenet 9), each pair's status and result, and the groups when it
ended — rewritten after every pair, so a stopped or killed job still says which pairs finished.

**Resumable by construction.** A pair already compared with these settings — either way round,
by this job or by hand — is skipped, so running the job again after a stop does only what is
left.

**A failed pair is a row, not the end.** A pair whose members cannot be measured is recorded
with its reason and the job goes on; a Cancel keeps every finished pair.

**On a remote target too** (§6 "Remote analysis"), which is what makes every pair of a large
run (hundreds of pairs, many hours on CPU) practical: the job re-sends the clustering run it
reads, so the skip check there sees every pair already placed here, and places its comparisons
beside the run here once its own run directory — the one thing a copy-back returns — is back.
Placement is the same idempotent step either way, run after every pair where the job runs.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

from detail.api import cluster_lookalikes as lookalikes
from detail.configs.cluster_lookalikes import (
    PAIRS_LABELLED,
    PROFILE_FIELDS,
    RETIRED_FIELDS,
    ClusterLookalikesConfig,
)
from detail.configs.cluster_profile import COMPARE_NODE
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

#: The job's own fields beside the config's: the pairs resolved at launch (``[a, b]``; a job
#: written before suggested names were cut carries a third item, ignored), the source run's
#: recordings (recorded as the run's ``recording_ids``), where it runs, identity.
_JOB_EXTRA = (
    ("job_id", str),
    ("pair_list", list),
    ("source_recording_ids", list),
    ("compute", object),
    ("created_at", str),
)
#: Extras a job written before they existed reads as ``None``.
_OPTIONAL_EXTRA = ("compute",)

#: A profile's own stages, as this job's phases name them, and where each sits inside its
#: pair's share of the bar: ``measuring`` has a fraction (most of the time is reading tracks);
#: the two fits have none and hold the bar where it got to (§6 rule 1).
_STAGES = {
    "sampling": ("measuring", 0.0),
    "measuring": ("measuring", 0.0),
    "fitting": ("fitting", 0.8),
    "context": ("fitting", 0.95),
}
_MEASURE_SHARE = 0.8


def _job_class():
    """The job dataclass: every ``ClusterLookalikesConfig`` field by name, so a field the
    config gains reaches the worker and the record without an edit here
    (``tests/configs/test_record_completeness.py``)."""
    names = [name for name, _type in _JOB_EXTRA]
    specs = list(_JOB_EXTRA) + [
        (name, object)
        for name in ClusterLookalikesConfig.model_fields
        if name not in names
    ]

    def to_dict(self) -> dict:
        """Every field as written to ``job.json``; ``compute`` as its dict, since the remote
        shepherd sets it to the cluster's ``ComputeTarget`` before writing the envelope.
        """
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        if hasattr(out.get("compute"), "to_dict"):
            out["compute"] = out["compute"].to_dict()
        return out

    @classmethod
    def from_dict(cls, d: dict):
        known = {f.name for f in dataclasses.fields(cls)}
        defaults = ClusterLookalikesConfig.model_fields
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
        "ClusterLookalikesJob",
        specs,
        namespace={"to_dict": to_dict, "from_dict": from_dict},
        module=__name__,
    )


ClusterLookalikesJob = _job_class()


@dataclass
class ClusterLookalikesProgressEvent:
    """Progress in the §16 analysis event's field names: the outer pair counts **pairs**
    finished (skipped ones count from the start), the inner pair the share of the running
    pair's comparison. ``eta_s`` is over the pairs this job compares, not the ones it
    skipped."""

    recordings_done: int
    recordings_total: int
    frames_done: int
    frames_total: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None
    unit_label: str = "pair"
    fold_units: bool = True


@register_analysis_module
class ClusterLookalikesModule(AnalysisModule):
    module_id = lookalikes.MODULE_ID
    config_model = ClusterLookalikesConfig
    id_prefix = "an"
    # Reads another analysis run, not a recording's run: nothing a recording must carry.
    requires: list[str] = []
    job_class = ClusterLookalikesJob
    phases = ("measuring", "fitting")
    progress_event_class = ClusterLookalikesProgressEvent
    # frames_total is a WITHIN_RESOLUTION fraction
    progress_counts_frames = False
    # §6 "Remote analysis": the job re-sends the clustering run it reads, and is placed beside
    # that run once it is back.
    remote_capable = True

    def remote_resources(self, job) -> dict:
        """One GPU, the fewest a job may ask for: the work is CPU, and Tillicum refuses a job
        that asks for none. The node's CPUs and memory scale down with it, so a target sized
        for an 8-GPU model run does not hold 8 idle GPUs for hours of pairs."""
        return {"gpus": 1}

    def refreshed_inputs(self, experiment_root, job) -> list[Path]:
        """The clustering run: its node cache, and the comparisons already placed there,
        which the job's skip check reads."""
        return [analysis_store.analysis_dir(experiment_root, job.source_run_id)]

    def after_copy_back(self, experiment_root, job, run_dir) -> None:
        """Place what the job compared on the cluster beside the clustering run here."""
        from detail.api import profile_describe
        from detail.api.experiment import Experiment

        profile_describe.place(Experiment(experiment_root), job.job_id)

    def validate(self, experiment_dir, config: dict) -> dict:
        """The pairs as they will run (Tenet 9): the branch tops — and, for ``pairs:
        labelled``, the run's labels — read now and resolved to a list of pairs, written back,
        so the job compares what it was launched for.

        Raises:
            ValueError: Not a clustering run, no node cache, no pair to compare (with why —
                for ``labelled``, no branch top labelled yet), a remote target not registered
                here, or another of these jobs already running on the run.
            FileNotFoundError: The run does not exist.
        """
        from detail.api import remote as api_remote
        from detail.api.experiment import Experiment

        given = dict(config)
        given.pop("source_recording_ids", None)  # read from the run, never chosen
        given.pop("pair_list", None)  # resolved here, never chosen
        for retired in RETIRED_FIELDS:  # an older run's settings, from "start from"
            given.pop(retired, None)
        compute = api_remote.analysis_compute(
            given.pop("compute", None), "find look-alike clusters"
        )
        model = ClusterLookalikesConfig.model_validate(given)
        experiment = Experiment(experiment_dir)
        run_id = model.source_run_id
        record = experiment.analysis_record(run_id)
        run = experiment._clustering_run(run_id)
        try:
            view = run.node_view()
        except FileNotFoundError:
            raise ValueError(
                f"{run_id} has no node cache yet; open it once (or rebuild its clips) "
                "before comparing its clusters"
            ) from None
        node_labels = dict(run.state().node_labels)
        if model.pairs == PAIRS_LABELLED and not lookalikes.labelled_tops(
            view, node_labels
        ):
            raise ValueError(
                f"{run_id} has no labelled cluster to compare: label at least one cluster "
                "first, or choose every pair (pairs: all)"
            )
        pairs = lookalikes.pairs_for(view, node_labels, model.pairs)
        if not pairs:
            tops = lookalikes.branch_tops(view)
            if len(tops) < 2:
                why = (
                    f"it has {len(tops)} branch tops (both halves of a split at least "
                    f"{view.min_cluster_s:g} s, the time a cluster needs to count as one worth naming)"
                )
            elif model.pairs == PAIRS_LABELLED and lookalikes.candidate_pairs(view):
                why = (
                    "every pair with a labelled side has both sides under one label, or "
                    "one inside the other; choose every pair (pairs: all)"
                )
            else:
                why = "every pair of its branch tops has one inside the other"
            raise ValueError(f"{run_id} has no pair of clusters to compare: {why}")
        _refuse_a_second_job(experiment, run_id)
        return {
            **model.model_dump(),
            "pair_list": [[int(a), int(b)] for a, b in pairs],
            "source_recording_ids": list(record.get("recording_ids") or []),
            "compute": compute,
        }

    def build_job(self, run_id: str, config: dict):
        fields = {f.name for f in dataclasses.fields(ClusterLookalikesJob)}
        values = {k: v for k, v in config.items() if k in fields}
        values.update(job_id=run_id, created_at=utc_timestamp())
        return ClusterLookalikesJob.from_dict(values)

    def run_dir_for(self, experiment_root, job):
        return analysis_store.analysis_dir(experiment_root, job.job_id)

    def output_manifest(self, run_id: str, config: dict) -> list[dict]:
        """The comparisons as the job wrote them, before placement; the placed copies are the
        clustering run's, listed per pair in ``pairs`` and grouped in ``groups``."""
        return [
            {
                "path": "profiles",
                "kind": "profiles",
                "label": "Pair comparisons (placed beside the run)",
            }
        ]

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        from detail.api.experiment import Experiment

        run_dir = Path(run_dir)
        experiment = Experiment(
            run_dir.parent.parent
        )  # <experiment>/analysis/<run_id>/
        entries, status = run_pairs(experiment, job, run_dir, progress_cb, stop_event)
        record = analysis_store.read_analysis_yaml(run_dir)
        record.update(_summary(job, entries))
        record["groups"] = _groups(experiment, job.source_run_id)
        record["status"] = status
        record["artifacts"] = self.output_manifest(job.job_id, {})
        # Every setting this run was launched with, derived from the job (Tenet 9).
        record["params"] = job_params(job)
        analysis_store.write_analysis_yaml(run_dir, record)


def _refuse_a_second_job(experiment, run_id: str) -> None:
    """Two of these jobs on one run would compare the same pairs twice; the second is
    refused while the first runs."""
    from detail.core.dataclasses import ACTIVE_STATUSES
    from detail.executor import rundir

    for ref in experiment.analysis_runs:
        if ref.module_id != lookalikes.MODULE_ID or ref.status not in ACTIVE_STATUSES:
            continue
        try:  # the job envelope names it from submit on; the record only once it runs
            source = rundir.read_envelope(experiment.analysis_run_dir(ref.id))[
                "job"
            ].get("source_run_id")
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if source == run_id:
            raise ValueError(
                f"{ref.id} is already comparing {run_id}'s clusters; wait for it or "
                "cancel it"
            )


def profile_settings(job) -> dict:
    """The profile settings every pair of ``job`` is compared with, by name."""
    return {name: getattr(job, name) for name in PROFILE_FIELDS}


def pair_config(job, a: int, b: int) -> dict:
    """The cluster profile config comparing ``a`` against ``b`` with place left out."""
    return {
        **profile_settings(job),
        "compare": COMPARE_NODE,
        "against": int(b),
        "leave_out_place": True,
    }


def compared_pairs(source_dir, job) -> set[tuple[int, int]]:
    """The pairs already compared with ``job``'s settings, either way round."""
    wanted = profile_settings(job)
    return {
        item.pair
        for item in lookalikes.read_evidence(source_dir)
        if lookalikes.same_settings(item.params, wanted)
    }


def _groups(experiment, run_id: str) -> list[dict]:
    """The run's look-alike groups as the job leaves them, for its record."""
    run = experiment._clustering_run(run_id)
    return [
        {"nodes": list(group.nodes), "name": group.name, "links": len(group.links)}
        for group in run.lookalike_groups()
    ]


def _summary(job, entries: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    alike = sum(1 for e in entries if e.get("looks_alike"))
    return {
        "source_run_id": job.source_run_id,
        "counts": counts,
        "looks_alike": alike,
        "pairs": entries,
    }


def _write_progress_record(run_dir: Path, job, entries: list[dict]) -> None:
    """The record as it stands, rewritten after each pair (atomically), so a run stopped or
    killed part-way still lists which pairs finished."""
    record = analysis_store.read_analysis_yaml(run_dir)
    record.update(_summary(job, entries))
    record["params"] = job_params(job)
    analysis_store.write_analysis_yaml(run_dir, record)


def _result(profile, file: str) -> dict:
    e, place = profile.explanation, profile.place_explanation
    out = {
        "accuracy": round(float(e.accuracy), 4),
        "floor": round(float(e.floor), 4),
        "looks_alike": bool(e.accuracy == e.accuracy and not e.separated),
        "file": file,
    }
    if place is not None:
        out.update(
            place_accuracy=round(float(place.accuracy), 4),
            place_floor=round(float(place.floor), 4),
        )
    return out


def run_pairs(experiment, job, run_dir, progress_cb, stop_event):
    """Compare each pair of ``job.pair_list``; ``(entries, status)``.

    ``entries`` is one dict per pair in order — ``{a, b, status, ...}`` with the
    result of a compared one — and ``status`` the run's: ``complete`` when every pair finished
    or was already compared, ``partial`` when some did and others failed or were not reached,
    ``failed`` when none did.
    """
    from detail.api import cluster_profile
    from detail.api import profile_describe

    run_id = job.source_run_id
    source_dir = experiment.analysis_run_dir(run_id)
    done = compared_pairs(source_dir, job) if job.skip_existing else set()
    entries = [
        {
            "a": int(a),
            "b": int(b),
            "status": "skipped" if (min(a, b), max(a, b)) in done else "pending",
        }
        for a, b, *_overlap in job.pair_list
    ]
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
        message = MESSAGE_SEPARATOR.join(
            part
            for part in (
                phase,
                f"pair {clock['done'] + 1} of {total}",
                f"#{entry['a']} vs #{entry['b']}",
                detail,
            )
            if part
        )
        emit(
            ClusterLookalikesProgressEvent(
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
        began = time.monotonic()

        def on_profile(stage, fraction, detail, _entry=entry):
            phase, share = _STAGES.get(stage, ("fitting", 0.95))
            if stage == "measuring":
                share = _MEASURE_SHARE * min(max(float(fraction), 0.0), 1.0)
            report(_entry, phase, share, detail)

        report(entry, "measuring", 0.0, "drawing members")
        try:
            profile = experiment.profile_cluster(
                run_id,
                entry["a"],
                pair_config(job, entry["a"], entry["b"]),
                save=False,
                progress=on_profile,
            )
        except OperationCancelled:
            entry["status"] = "cancelled"
            stopped = True
            break
        except Exception as exc:  # noqa: BLE001 — one pair's isolation boundary
            entry.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                seconds=round(time.monotonic() - began, 2),
            )
        else:
            # Into this job's own run directory, then placed beside the run, where the
            # groups and the Interpret tab read it.
            cluster_profile.save_profile(run_dir, profile)
            profile_describe.place(experiment, job.job_id, [entry["a"]])
            folder = cluster_profile.profile_dir(source_dir, entry["a"], entry["b"])
            file = (folder / cluster_profile.PROFILE_FILENAME).relative_to(source_dir)
            entry.update(
                status="complete",
                seconds=round(time.monotonic() - began, 2),
                **_result(profile, file.as_posix()),
            )
        clock["done"] += 1
        clock["work"] += 1
        _write_progress_record(run_dir, job, entries)

    for entry in entries:
        if entry["status"] == "pending":
            entry["status"] = "not run"
    emit(
        ClusterLookalikesProgressEvent(
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
    elif finished or stopped:
        status = "partial"
    else:
        status = "failed"
    return entries, status
