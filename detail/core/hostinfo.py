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

**The same machine, for a bug report.** :func:`system_report` is the first section of
``detail doctor``: these facts plus what a fixer on another machine needs and cannot look up —
the exact build, the Python environment, which CUDA the PyTorch build targets beside the
newest one the driver supports, and every package detail runs with. Its rows are pasted into
public issues, so none of them carries a folder path or a user name. It adds no key to
:func:`collect`, whose dict is the ``machine.yaml`` a run writes (``DESIGN_PLAN.md`` §6).

**One ``nvidia-smi`` reader.** :func:`nvidia_smi` serves this module and the GUI's CUDA doctor
(``gui/cuda_doctor.py``), which used to parse the same header with its own copy.
"""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import platform
import re
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


_NVIDIA_SMI_DRIVER = re.compile(r"Driver Version:\s*([\d.]+)")
_NVIDIA_SMI_CUDA = re.compile(r"CUDA Version:\s*([\d.]+)")


def parse_nvidia_smi(stdout: str) -> tuple[str | None, str | None]:
    """``(driver version, newest CUDA the driver supports)`` from ``nvidia-smi``'s header."""
    driver = _NVIDIA_SMI_DRIVER.search(stdout)
    cuda = _NVIDIA_SMI_CUDA.search(stdout)
    return (driver.group(1) if driver else None, cuda.group(1) if cuda else None)


def nvidia_smi() -> tuple[str | None, str | None]:
    """``(driver version, newest CUDA the driver supports)``, or ``(None, None)``.

    Read from ``nvidia-smi`` because torch exposes neither. Absent on every non-NVIDIA machine,
    which is not a failure. The CUDA the driver reports is a ceiling, not the version in use:
    a PyTorch build for any CUDA at or below it runs.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None, None
    try:
        out = subprocess.run(
            [exe],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None  # a binary that could not start, or one that hung
    if out.returncode != 0:
        return None, None
    return parse_nvidia_smi(out.stdout or "")


def _driver() -> str:
    """The NVIDIA driver version, which is the field a timing difference most often turns on."""
    driver, _ = nvidia_smi()
    return driver or UNKNOWN


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


#: Extras that install tools for working on detail, not anything detail runs with.
_TOOLING_EXTRAS = frozenset({"dev", "docs"})
#: Installed outside detail's own metadata: SAM 3 is a git clone, because Meta ships it off PyPI.
_UNDECLARED_PACKAGES = ("sam3",)
_REQUIREMENT_NAME = re.compile(r"[A-Za-z0-9._-]+")
_MARKER_EXTRA = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")


def package_versions() -> list[tuple[str, str]]:
    """``(name, version)`` for each installed package detail runs with, in declaration order.

    Read from detail's own package metadata, so a dependency added to ``pyproject.toml`` is
    reported the day it is added; a list kept here would drift from it. The ``dev`` and
    ``docs`` extras are left out, and so is anything not installed: an extra nobody installed
    is not news, and its absence is what ``detail doctor``'s checks are for.
    """
    try:
        declared = metadata.requires("detail") or []
    except metadata.PackageNotFoundError:
        declared = []  # running from a source tree that was never installed
    names: list[str] = []
    for requirement in declared:
        match = _REQUIREMENT_NAME.match(requirement)
        if match is None:
            continue
        _, _, marker = requirement.partition(";")
        extras = set(_MARKER_EXTRA.findall(marker))
        if extras and extras <= _TOOLING_EXTRAS:
            continue
        if match.group(0) not in names:
            names.append(match.group(0))
    names.extend(name for name in _UNDECLARED_PACKAGES if name not in names)
    versions = []
    for name in names:
        try:
            versions.append((name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            continue
    return versions


def _os_name() -> str:
    """The OS and its version; on Linux the distribution too, since `platform` names only the
    kernel and C library."""
    name = platform.platform()
    if sys.platform.startswith("linux"):
        try:
            pretty = platform.freedesktop_os_release().get("PRETTY_NAME")
        except OSError:
            pretty = None
        if pretty:
            name = f"{pretty} ({name})"
    return name


def _environment_kind() -> str:
    """Which kind of Python environment this is. Never its path."""
    if (Path(sys.prefix) / "conda-meta").is_dir():
        return "conda environment"
    if sys.prefix != sys.base_prefix:
        return "virtual environment"
    return "system Python, no environment"


def _gb(value) -> str:
    """A memory size for a row: ``64.0 GB``, or the word the probe gave back."""
    return f"{value} GB" if isinstance(value, (int, float)) else str(value)


def _gpu_line(facts: dict) -> str:
    """Every GPU PyTorch can use; the first with its memory, from :func:`collect`."""
    if facts["accelerator"] == "cuda":
        from detail.core.utils import cuda_device_names

        line = f"{facts['gpu']} ({_gb(facts['gpu_memory_gb'])})"
        others = cuda_device_names()[1:]
        return f"{line}, plus {', '.join(others)}" if others else line
    if facts["accelerator"] == "mps":
        return f"{facts['gpu']}, through MPS"
    return "none that PyTorch can use"


def _driver_line(driver: str | None, driver_cuda: str | None) -> str:
    if driver is None:
        return "none found (nvidia-smi is missing or did not answer)"
    if driver_cuda is None:
        return driver
    return f"{driver}, supports CUDA up to {driver_cuda}"


def _torch_line(version: str) -> str:
    """The PyTorch version and what it was built for: CUDA, ROCm, or neither.

    The build is the half that decides whether a GPU is usable: a CPU-only wheel on an NVIDIA
    machine reports no GPU at all, and only this line beside the driver's says why.
    """
    if version == UNKNOWN:
        return "not installed, or fails to import"
    try:
        import torch
    except Exception:  # noqa: BLE001 — a report must survive a broken install
        return version
    cuda = getattr(torch.version, "cuda", None)
    hip = getattr(torch.version, "hip", None)
    if cuda:
        return f"{version}, built for CUDA {cuda}"
    if hip:
        return f"{version}, built for ROCm {hip}"
    return f"{version}, built without CUDA"


def system_report() -> list[tuple[str, str]]:
    """This machine and install as ``(label, value)`` rows: ``detail doctor``'s first section.

    What a bug report needs and a fixer on another machine cannot look up. No value carries a
    folder path or a user name, because the rows are made to be pasted into a public issue.
    Imports torch, which a ``detail doctor`` process can afford and a pipeline job must not.
    """
    from detail.core import buildinfo

    facts = collect()
    driver, driver_cuda = nvidia_smi()
    packages = ", ".join(f"{name}=={version}" for name, version in package_versions())
    return [
        ("Build", buildinfo.build_info().describe()),
        ("OS", _os_name()),
        ("Python", f"{platform.python_version()}, {_environment_kind()}"),
        ("CPU", f"{facts['cpu']}, {facts['cpu_cores']} cores"),
        ("RAM", _gb(facts["ram_gb"])),
        ("GPU", _gpu_line(facts)),
        ("NVIDIA driver", _driver_line(driver, driver_cuda)),
        ("PyTorch", _torch_line(facts["torch"])),
        ("Packages", packages or "none found in detail's install metadata"),
    ]
