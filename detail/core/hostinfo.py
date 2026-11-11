"""What machine this is — the half of a run's performance report that is not about the run.

**Why a run's timings are worth little on their own.** Three hours of detection is fast or slow
depending on what it ran on, and the people who need that comparison are not in one lab. A
million frames might take ten hours on an older consumer GPU, two on a current one, and something
else again on a laptop; what is missing is a shared, community-sourced reference for how long
detection, tracking and embedding actually take on each kind of machine.

So a report carries the box beside the numbers, and a person can paste both.

**Local only. Nothing is sent anywhere.** That is a deliberate design line, and this module has
no network call in it — the report is a file, and sharing it is somebody deciding to.

**RAM needs `psutil`, and `psutil` is a core dependency (#394, OQ-RUN-9).** `platform` and
`torch` give the OS, the core count, the torch build, and on an NVIDIA box the GPU's name and
memory; `psutil` gives RAM. It used to live only in the `sam3` extra, which meant a plain
install always reported RAM as "unknown" — blank exactly when the report is most needed, to
compare machines. Now a missing `psutil` means something actually went wrong with the install
(`detail doctor` flags it), not that this machine was never asked — the report still says
"unknown" rather than guessing, but that word now means "this machine would not tell us."

Every probe is individually guarded. A report is a diagnostic, and one that raises because a
driver is unusual has failed at the only job it had.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

#: What a field says when the machine would not tell us. Not `""` and not `0`: a blank reads as
#: "nobody filled this in" and a zero reads as a measurement, and a reader comparing two reports
#: needs to know which of those it is looking at.
UNKNOWN = "unknown"


#: What the torch-derived fields say for a job that never loaded torch. Not `UNKNOWN`: the
#: machine was not asked, because asking means importing torch, which a CPU-only job must not do.
NOT_LOADED = "not loaded by this run"


def _torch_facts(load_torch: bool = True) -> dict:
    """GPU name, memory, and the torch build — or what we could get.

    Imported here rather than at module scope: importing torch costs seconds, and this module is
    read by a report, not by the pipeline. With ``load_torch=False`` it reads only a torch that is
    already imported — the form a job's own snapshot uses, so a CPU-only job stays torch-free.
    """
    facts = {
        "torch": UNKNOWN,
        "accelerator": "cpu",
        "gpu": UNKNOWN,
        "gpu_memory_gb": UNKNOWN,
    }
    if not load_torch and "torch" not in sys.modules:
        return {key: NOT_LOADED for key in facts}
    try:
        import torch
    except Exception:  # noqa: BLE001 — a report must survive a broken install
        return facts
    # `str()`: torch's version is a `TorchVersion`, a str subclass that YAML writes as a Python
    # object tag, which the safe loader then refuses — so the machine file read back as missing.
    facts["torch"] = str(getattr(torch, "__version__", UNKNOWN))
    try:
        if torch.cuda.is_available():
            facts["accelerator"] = "cuda"
            facts["gpu"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            facts["gpu_memory_gb"] = round(total / 2**30, 1)
        elif torch.backends.mps.is_available():
            # An Apple GPU has no separate memory to report — it shares the machine's, which the
            # RAM row already gives. Saying "unknown" there would read as a failed probe.
            facts["accelerator"] = "mps"
            facts["gpu"] = f"Apple {platform.machine()} (unified memory)"
            facts["gpu_memory_gb"] = "shared"
    except Exception:  # noqa: BLE001
        pass
    return facts


def _driver() -> str:
    """The NVIDIA driver version, which is the field a timing difference most often turns on.

    Read from `nvidia-smi` because torch does not expose it. Absent on every non-NVIDIA machine,
    which is not a failure.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return UNKNOWN
    try:
        out = subprocess.run(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return UNKNOWN
    first = (out.stdout or "").strip().splitlines()
    return first[0].strip() if first else UNKNOWN


def _ram_gb():
    """Total RAM, via `psutil` (a core dependency, #394).

    Guarded the same as every other probe here: a broken or half-finished install is not this
    module's problem to raise about, so a missing `psutil` still degrades to `UNKNOWN` rather
    than crashing the report — `detail doctor` is where that install gets flagged. Reported as
    unknown rather than guessed either way — a report whose numbers are sometimes invented is
    worse than one with a gap.
    """
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return UNKNOWN
    try:
        return round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:  # noqa: BLE001
        return UNKNOWN


def collect(load_torch: bool = True) -> dict:
    """This machine, as a flat dict of strings and numbers ready to write.

    Flat because it is written to YAML, shown in a table, and pasted into an issue; a nested
    shape would have to be flattened at each of those. ``load_torch=False`` never imports torch
    (see :func:`_torch_facts`).
    """
    from detail import __version__

    facts = {
        "detail": __version__,
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or UNKNOWN,
        "cpu_cores": os.cpu_count() or UNKNOWN,
        "ram_gb": _ram_gb(),
        "driver": _driver(),
    }
    facts.update(_torch_facts(load_torch))
    return facts


def as_lines(facts: dict) -> list[str]:
    """The machine block of the report, in the order a reader scans it.

    GPU first, because it is what a timing comparison turns on, and the reason someone is reading
    this at all.
    """
    order = [
        ("GPU", "gpu"),
        ("GPU memory (GB)", "gpu_memory_gb"),
        ("Driver", "driver"),
        ("Accelerator", "accelerator"),
        ("CPU", "cpu"),
        ("CPU cores", "cpu_cores"),
        ("RAM (GB)", "ram_gb"),
        ("OS", "os"),
        ("Machine", "machine"),
        ("detail", "detail"),
        ("torch", "torch"),
    ]
    return [f"| {label} | {facts.get(key, UNKNOWN)} |" for label, key in order]
