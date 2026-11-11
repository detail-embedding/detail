"""Detached coordinator for a durable distill run (§12).

Invoked as ``python -m detail.executor.distill_coordinator <run_dir>`` where ``run_dir``
is ``<experiment>/distill/<run_id>/``. Spawned in its own process group (like the per-job
worker and the processing-run coordinator), so it survives the launching app closing: it
reads the reserved ``distill.yaml``, sequences the two stages via ``DistillOrchestrator``
(``label_dataset`` then ``train_model``), and publishes the folded ``DistillProgress`` to
``distill/<run_id>/progress.json`` that any process reattaches to.

Isolation boundary: it always drives a terminal ``distill.yaml`` — ``DistillOrchestrator``
finalizes on any stage failure, and this wrapper is the backstop for a failure before the
loop. If even that fails, the process exits and reconcile-on-open finalizes the run (the
coordinator pid reads dead).
"""

import contextlib
import os
import sys
from pathlib import Path

from detail.api.experiment import DistillOrchestrator, Experiment
from detail.core.dataclasses import ACTIVE_STATUSES, distill_progress_to_dict
from detail.executor.runners import RUNNER_MODULES_ENV, import_runner_modules
from detail.storage import activity
from detail.storage import distill_runs as distill_store


def _mark_terminal(experiment_root, run_id: str, status: str, error: str) -> None:
    """Best-effort terminal status if the orchestrator never got to finalize;
    reconcile-on-open is the ultimate backstop, so failures here are swallowed."""
    with contextlib.suppress(Exception):
        record = distill_store.read_run(experiment_root, run_id)
        if record.get("status") in ACTIVE_STATUSES:
            record["status"] = status
            record.setdefault("error", error)
            distill_store.write_run(experiment_root, run_id, record)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(
            "usage: python -m detail.executor.distill_coordinator <run_dir>"
        )
    run_dir = Path(argv[0])
    run_id = run_dir.name
    experiment_root = run_dir.parents[
        1
    ]  # <experiment>/distill/<run_id> -> <experiment>
    modules = [m for m in os.environ.get(RUNNER_MODULES_ENV, "").split(",") if m]
    import_runner_modules(modules)
    try:
        record = distill_store.read_run(experiment_root, run_id)
        experiment = Experiment.open(experiment_root)

        # The third writer. A distill's units are its two stages, so `unit` carries the phase
        # name and `phase` its message — the same columns, read by the same reader, rather than
        # a distill-shaped log nothing else understands.
        log = activity.ActivityLog(distill_store.run_dir(experiment_root, run_id))

        def emit(p) -> None:
            distill_store.write_progress(
                experiment_root, run_id, distill_progress_to_dict(p)
            )
            log.sample(
                activity.make_row(
                    elapsed_s=p.elapsed_s,
                    unit=p.phase,
                    unit_index=p.stage_index + 1,
                    unit_total=p.stage_count,
                    phase=p.message or "",
                    done=round(p.inner_fraction * 1000),
                    total=1000,
                    eta_unit_s=p.eta_s,
                    eta_run_s=p.eta_s,
                )
            )

        DistillOrchestrator(experiment, run_id, record, emit).run()
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
