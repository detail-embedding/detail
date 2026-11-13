"""Experiment-level analysis-run records: ``analysis/<run_id>/analysis.yaml`` (§16).

An *analysis run* applies one analysis module over a selection of recordings
(``Experiment.run_analysis``). It is a single experiment-level job: its run directory
``analysis/<run_id>/`` holds the executor envelope (``job.json``), the per-recording
output artifacts, and ``analysis.yaml`` — the record (full lineage + artifact
manifest, ``schema_version`` first). Runs are discovered by **scanning**
``analysis/`` — no central mutable list (the same pattern as datasets/models and
recordings).

``analysis.yaml`` is written **atomically** (sibling ``*.tmp`` + ``os.replace``):
the runner subprocess rewrites it on completion while the GUI's progress view
reads it, so a plain truncating write would let a reader see a torn/empty file
(atomic-writes tenet — the same reason ``storage/runs.py`` is atomic; the
datasets/models stores predate the rule and are borderline, not a template here).
"""

import contextlib
import shutil
from pathlib import Path

import yaml

from detail.core import atomic
from detail.core import yaml_load
from detail.storage import dirscan
from detail.storage import experiment as experiment_store
from detail.storage import read_cache

SCHEMA_VERSION = 1
ANALYSIS_FILENAME = "analysis.yaml"


def analysis_dir(experiment_dir: Path | str, run_id: str) -> Path:
    return experiment_store.analysis_dir(experiment_dir) / run_id


def discover_analyses(experiment_dir: Path | str) -> list[str]:
    """Sorted ids of analysis runs (subdirs of ``analysis/`` holding analysis.yaml)."""
    root = experiment_store.analysis_dir(experiment_dir)
    return dirscan.discover_ids(root, ANALYSIS_FILENAME)


def _highest_claimed(experiment_dir: Path | str) -> int:
    """The highest ``an_NN`` number **claimed**, whether or not it has a record yet.

    Deliberately not ``discover_analyses``, which counts only directories holding an
    ``analysis.yaml`` — the right definition of "an analysis run" for listing, the wrong one
    for allocating: a directory reserved a moment ago has no record yet, and skipping it hands
    the same id out again (and then every retry loses to it).
    """
    root = experiment_store.analysis_dir(experiment_dir)
    nums = []
    for path in dirscan.child_dirs(root):
        try:
            nums.append(int(path.name.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return max(nums, default=0)


def reserve_analysis_id(experiment_dir: Path | str, attempts: int = 50) -> str:
    """Claim the next ``an_NN`` id by creating its directory — safe under concurrency.

    ``next_analysis_id`` scans and returns max+1, which is a read: two analysis runs launched
    at once both see the same highest id and both take it, and the second overwrites the
    first's record — a wrong answer rather than an error. Creating the directory with
    ``exist_ok=False`` makes the claim atomic (``mkdir`` either creates or raises), so a loser
    rescans and takes the next one. The same idiom the removed
    ``clusterers.reserve_clusterer_id`` used, where this hazard was first solved.

    Raises:
        RuntimeError: If ``attempts`` consecutive claims are all lost — far more contention
            than a real workload has, and worth surfacing rather than spinning.
    """
    root = Path(experiment_dir)
    experiment_store.analysis_dir(root).mkdir(parents=True, exist_ok=True)
    for _ in range(attempts):
        run_id = f"an_{_highest_claimed(root) + 1:02d}"
        try:
            analysis_dir(root, run_id).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return run_id
    raise RuntimeError(
        f"could not reserve an analysis id under {root} after {attempts} attempts — "
        "too many analysis runs are starting at once"
    )


def next_analysis_id(experiment_dir: Path | str) -> str:
    """Experiment-monotonic ``an_NN`` id, one past the highest existing (matches
    ``AnalysisModule.id_prefix``).

    **Reads only.** Two callers racing here get the same id, so anything that goes on to
    create the run must use :func:`reserve_analysis_id`; this remains for callers that only
    need to *predict* the next id.
    """
    nums = []
    for rid in discover_analyses(experiment_dir):
        try:
            nums.append(int(rid.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return f"an_{max(nums, default=0) + 1:02d}"


def write_analysis_yaml(run_dir: Path | str, record: dict) -> None:
    """Write ``analysis.yaml`` with ``schema_version`` as the first key (Tenet 7).

    Atomic (tmp + ``os.replace``) — the runner subprocess rewrites this on
    completion while the GUI detail page reads it; a reader never sees a torn file.
    """
    rdir = Path(run_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    ordered = {"schema_version": SCHEMA_VERSION}
    for key, value in record.items():
        if key != "schema_version":
            ordered[key] = value
    path = rdir / ANALYSIS_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(ordered, f, sort_keys=False)
    read_cache.replace_document(tmp, path)


def read_analysis_yaml(run_dir: Path | str) -> dict:
    """Read one analysis-run record.

    Raises:
        FileNotFoundError: If the record is absent.
    """
    path = Path(run_dir) / ANALYSIS_FILENAME
    try:
        return read_cache.cached_document(
            path, lambda: yaml_load.safe_load(atomic.read_text(path))
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"no {ANALYSIS_FILENAME} in {run_dir}") from None


def set_analysis_status(
    run_dir: Path | str, status: str, error: str | None = None
) -> None:
    """Update the ``status`` field of an existing ``analysis.yaml``, in place.

    ``error`` (a failure message) is recorded so the GUI detail page can show why
    a run failed. Also merges in ``duration_s`` from the job's own envelope (§ run
    location, duration, cost) — analysis runs are local-only today, so no
    ``compute``/cost, just timing.
    """
    from detail.executor import run_metrics

    record = read_analysis_yaml(run_dir)
    record["status"] = status
    if error is not None:
        record["error"] = error
    with contextlib.suppress(Exception):
        record.update(run_metrics.completion_fields(run_dir, None))
    write_analysis_yaml(run_dir, record)


#: Set on an authored run whose §19 artifact is not known to match its authored state.
#:
#: An authored edit rewrites three things in order — the state, the history, the artifact —
#: and the first two are durable before the third is attempted. A reprojection that dies (a
#: transient network-share write, a kill) therefore leaves ``temporal_clustering.yaml`` saying
#: ``granularity: 138`` while ``behavior_output.yaml`` says ``null``, with a history line
#: asserting the change succeeded and nothing anywhere recording that they disagree. §19's
#: rule for a materialized view is to regenerate rather than reconcile, so this is the flag
#: that says it needs regenerating; ``AuthoredRun.reproject`` clears it.
STALE_KEY = "artifact_stale"


def artifact_stale(run_dir: Path | str) -> bool:
    """Whether this run's artifact is known not to match its authored state."""
    return bool(read_analysis_yaml(run_dir).get(STALE_KEY))


def set_artifact_stale(run_dir: Path | str, stale: bool) -> None:
    """Record (or clear) the artifact-vs-state disagreement, in place."""
    record = read_analysis_yaml(run_dir)
    if bool(record.get(STALE_KEY)) == stale:
        return
    record[STALE_KEY] = stale
    write_analysis_yaml(run_dir, record)


def delete_analysis(experiment_dir: Path | str, run_id: str) -> None:
    """Remove an analysis run's directory and everything it owns. Idempotent — a
    missing directory is a no-op (the store owns its dir ops, like runs)."""
    shutil.rmtree(analysis_dir(experiment_dir, run_id), ignore_errors=True)
