"""What a job ran on and how much memory it used, kept in the job's own run directory.

**Why at run time.** A performance report used to describe the machine doing the *export*, so a
run moved to another computer — or exported from a laptop after it ran on a GPU box — reported
the wrong hardware beside its timings. The job now writes the machine it ran on, and the report
reads that.

Both files are written once, by the worker, when the job ends (``executor/worker.py``), and read
by the Performance report (``api/export_targets.py``). A run from before they existed has
neither; a reader gets ``None`` and says so rather than guessing.
"""

from __future__ import annotations

from pathlib import Path

from detail.core import hostinfo
from detail.core import progress as core_progress
from detail.core import resource_peaks
from detail.storage import state_yaml

MACHINE_FILENAME = "machine.yaml"
RESOURCES_FILENAME = "resources.yaml"


@core_progress.best_effort("run facts")
def record(run_dir: Path | str, sampler: "resource_peaks.PeakSampler") -> None:
    """Stop ``sampler`` and write both files. Best-effort: a report about the job must never
    fail the job, so the probes and the writes all sit inside the guard.

    Read at the end rather than the start, so a GPU job's torch is already loaded and is
    described, while a CPU-only job is not made to import one (``load_torch=False``).
    """
    resources = sampler.stop()
    machine = hostinfo.collect(load_torch=False)
    state_yaml.write_state_yaml(Path(run_dir) / MACHINE_FILENAME, machine)
    state_yaml.write_state_yaml(Path(run_dir) / RESOURCES_FILENAME, resources)


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return state_yaml.read_state_yaml(path)
    except Exception:  # noqa: BLE001 — a torn or foreign file reads as "not recorded"
        return None


def read_machine(run_dir: Path | str) -> dict | None:
    """The machine the job ran on, or ``None`` if it did not record one."""
    return _read(Path(run_dir) / MACHINE_FILENAME)


def read_resources(run_dir: Path | str) -> dict | None:
    """The job's peak memory, or ``None`` if it did not record any."""
    return _read(Path(run_dir) / RESOURCES_FILENAME)
