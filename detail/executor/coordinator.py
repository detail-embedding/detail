"""Detached run coordinator for a durable multi-recording processing run (§15).

Invoked as ``python -m detail.executor.coordinator <run_dir>`` where ``run_dir``
is ``<experiment>/runs/<run_id>/``. Spawned in its own process group (like the per-job
worker), so it survives the launching app closing: it reads the reserved ``run.yaml``,
sequences the per-recording workers via ``RunOrchestrator``, and publishes aggregate
``ProcessingProgress`` to ``runs/<run_id>/progress.json`` that any process reattaches
to. This is the higher, *run-level* coordinator — distinct from the *part-level*
coordinator inside a single detection/embedding worker.

Isolation boundary: it always drives a terminal ``run.yaml`` — ``RunOrchestrator``
finalizes on any in-loop failure, and this wrapper is the backstop for a failure
before the loop. If even that fails, the process exits and reconcile-on-open
finalizes the run (the coordinator pid reads dead).
"""

import contextlib
import os
import sys
from pathlib import Path

from detail.api.experiment import Experiment, RunOrchestrator
from detail.core.dataclasses import ACTIVE_STATUSES, processing_progress_to_dict
from detail.executor.runners import RUNNER_MODULES_ENV, import_runner_modules
from detail.storage import runs as runs_store


def _mark_terminal(experiment_root, run_id: str, status: str, error: str) -> None:
    """Best-effort terminal status if the orchestrator never got to finalize;
    reconcile-on-open is the ultimate backstop, so failures here are swallowed."""
    with contextlib.suppress(Exception):
        record = runs_store.read_run(experiment_root, run_id)
        if record.get("status") in ACTIVE_STATUSES:
            record["status"] = status
            record.setdefault("error", error)
            runs_store.write_run(experiment_root, run_id, record)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: python -m detail.executor.coordinator <run_dir>")
    run_dir = Path(argv[0])
    run_id = run_dir.name
    experiment_root = run_dir.parents[1]  # <experiment>/runs/<run_id> -> <experiment>
    modules = [m for m in os.environ.get(RUNNER_MODULES_ENV, "").split(",") if m]
    import_runner_modules(modules)
    try:
        record = runs_store.read_run(experiment_root, run_id)
        experiment = Experiment.open(experiment_root)

        def emit(p) -> None:
            runs_store.write_progress(
                experiment_root, run_id, processing_progress_to_dict(p)
            )

        RunOrchestrator(
            experiment,
            run_id,
            record["module_id"],
            record["recording_ids"],
            record["config"],
            emit,
            record.get("created_at", ""),
            # Seed the prior results so a retry (Experiment.retry_run re-spawns us on the
            # same record) drives only the recordings that aren't complete. Empty on a
            # fresh run, where every recording is pending.
            per_recording=record.get("per_recording") or {},
        ).run()
    except KeyboardInterrupt:
        _mark_terminal(experiment_root, run_id, "cancelled", "interrupted (SIGINT)")
        return 1
    except Exception as exc:  # noqa: BLE001 — backstop; never leave the run "running"
        _mark_terminal(
            experiment_root,
            run_id,
            "failed",
            f"coordinator error: {type(exc).__name__}: {exc}",
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
