"""CUDA setup diagnostics for the GPU / CUDA page (§1) — the "CUDA doctor".

Qt-free so the probe/diagnosis are unit-testable. ``probe`` runs off the UI thread (it
imports torch lazily, Tenet 13, and shells out to ``nvidia-smi``); ``diagnose`` is pure.
The page renders the ``Diagnosis``. ``nvidia-smi`` is a diagnostic, not pipeline code, so the
FFmpeg-subprocess rule (Tenet 15) does not apply; it is read through ``core/hostinfo``, which
``detail doctor`` reads too.
"""

from dataclasses import dataclass

from detail.core import hostinfo
from detail.core.utils import cuda_available, cuda_device_names, mps_available

# PyTorch's CUDA wheel tags, newest first. The suggestion is the newest tag at or below the
# CUDA version the driver reports, which is a maximum, not the version to match. Read off
# https://download.pytorch.org/whl/torch/ on 2026-09-30; cu117 and older stop at torch 2.0.
# A tag missing here is never suggested, so re-read the index when PyTorch adds one.
# docs/getting-started.md's GPU and CUDA section follows this list.
_CUDA_WHEELS = ("132", "130", "129", "128", "126", "124", "121", "118")
NVIDIA_DRIVER_URL = "https://www.nvidia.com/download/index.aspx"
PYTORCH_SELECTOR_URL = "https://pytorch.org/get-started/locally/"
# Shown under every fix command: the app imported torch to diagnose it, and a running
# process keeps the compiled extension it loaded whatever is reinstalled underneath it.
RESTART_AFTER_FIX = (
    "Restart DETAIL after the install finishes. A running app keeps the PyTorch it "
    "started with, so this page shows the old build until then."
)


@dataclass(frozen=True)
class ProbeResult:
    """Raw machine facts for the CUDA doctor (gathered off the UI thread)."""

    cuda: bool  # torch.cuda.is_available()
    mps: bool
    cuda_devices: list[str]
    torch_installed: bool
    torch_cuda_build: str | None  # torch.version.cuda; None = a build with no CUDA
    driver_version: str | None  # from nvidia-smi; None = no NVIDIA driver/GPU
    driver_cuda: str | None  # the max CUDA the driver supports (nvidia-smi header)


@dataclass(frozen=True)
class Diagnosis:
    """What the GPU / CUDA page shows: a headline + diagnostics + an optional fix."""

    headline: str
    level: str  # "ready" | "warn" | "error"
    compatible: (
        bool  # an NVIDIA GPU is present → the doctor panel is enabled (not grey)
    )
    checks: list[tuple[str, bool, str]]  # (label, ok, detail)
    problem: str | None
    fix_command: str | None
    links: list[tuple[str, str]]  # (label, url)
    fix_reason: str | None  # why this wheel: the driver's CUDA → the tag it can run


def probe() -> ProbeResult:
    """Gather the machine's CUDA facts. Off the UI thread — imports torch lazily and
    shells out to ``nvidia-smi``."""
    driver_version, driver_cuda = hostinfo.nvidia_smi()
    try:
        import torch

        torch_installed = True
        torch_cuda_build = torch.version.cuda  # None on a build with no CUDA
    except ImportError:
        torch_installed = False
        torch_cuda_build = None
    return ProbeResult(
        cuda=cuda_available(),
        mps=mps_available(),
        cuda_devices=cuda_device_names(),
        torch_installed=torch_installed,
        torch_cuda_build=torch_cuda_build,
        driver_version=driver_version,
        driver_cuda=driver_cuda,
    )


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _cuda_wheel_for(driver_cuda: str | None) -> str | None:
    """The newest torch wheel tag (``cuXXX``) the driver's CUDA supports, or None."""
    if driver_cuda is None:
        return None
    try:
        major, minor = _version_tuple(driver_cuda)[:2]
    except ValueError:
        return None
    driver_tag = major * 10 + minor  # 12.8 -> 128
    for wheel in _CUDA_WHEELS:  # newest first
        if int(wheel) <= driver_tag:
            return f"cu{wheel}"
    return None


def _wheel_reason(driver_cuda: str | None, wheel: str | None) -> str | None:
    """The reasoning behind the suggested wheel, in the user's own numbers."""
    if driver_cuda is None:
        return None
    if wheel is None:
        return (
            f"Your driver supports up to CUDA {driver_cuda}, older than the oldest "
            f"PyTorch build listed here (cu{_CUDA_WHEELS[-1]}). Update your NVIDIA "
            "driver first."
        )
    return (
        f"Your driver supports up to CUDA {driver_cuda}, so the newest PyTorch build "
        f"it can run is {wheel}."
    )


def _reinstall_command(wheel: str) -> str:
    return (
        "uv pip install --reinstall-package torch --reinstall-package torchvision "
        f"torch torchvision --index-url https://download.pytorch.org/whl/{wheel}"
    )


def _is_newer(a: str | None, b: str | None) -> bool:
    """Whether CUDA version ``a`` is newer than ``b`` (e.g. '12.8' > '12.1')."""
    if a is None or b is None:
        return False
    try:
        return _version_tuple(a) > _version_tuple(b)
    except ValueError:
        return False


def _checks(p: ProbeResult) -> list[tuple[str, bool, str]]:
    if p.driver_version is not None:
        driver_detail = f"{p.driver_version} (max CUDA {p.driver_cuda})"
    else:
        driver_detail = "not found"
    if p.torch_cuda_build is not None:
        build_detail = f"CUDA {p.torch_cuda_build}"
    else:
        build_detail = "not installed" if not p.torch_installed else "no CUDA build"
    return [
        ("NVIDIA driver", p.driver_version is not None, driver_detail),
        ("PyTorch CUDA build", p.torch_cuda_build is not None, build_detail),
        ("PyTorch sees the GPU", p.cuda, "yes" if p.cuda else "no"),
    ]


def diagnose(p: ProbeResult) -> Diagnosis:
    """Map the machine facts to a headline, diagnostics, and (when fixable) a command."""
    compatible = p.driver_version is not None or p.cuda
    checks = _checks(p)
    links = [
        ("NVIDIA driver download ↗", NVIDIA_DRIVER_URL),
        ("PyTorch install selector ↗", PYTORCH_SELECTOR_URL),
    ]

    if p.cuda:
        devices = ", ".join(p.cuda_devices) or "CUDA device"
        return Diagnosis(
            f"✓ CUDA working: {devices}. Runs default to CUDA.",
            "ready",
            True,
            checks,
            None,
            None,
            [],
            None,
        )
    if not p.torch_installed:
        return Diagnosis(
            "PyTorch isn't installed.",
            "error",
            compatible,
            checks,
            "DETAIL needs PyTorch to use a GPU. Install the torch extra.",
            "uv pip install -e '.[torch]'",
            links,
            None,
        )
    if p.driver_version is not None:  # an NVIDIA GPU is present, but torch can't use it
        wheel = _cuda_wheel_for(p.driver_cuda)
        if p.torch_cuda_build is None:
            problem = (
                "PyTorch is a CPU-only build — reinstall a CUDA build to use the GPU."
            )
        elif _is_newer(p.torch_cuda_build, p.driver_cuda):
            problem = (
                f"PyTorch's CUDA ({p.torch_cuda_build}) is newer than your driver "
                f"supports (CUDA {p.driver_cuda}) — update your driver, or install a "
                "torch build for your driver's CUDA."
            )
        else:
            problem = "CUDA is present but PyTorch can't initialize it — update your NVIDIA driver."
        return Diagnosis(
            "⚠ NVIDIA GPU found, but PyTorch can't use it.",
            "warn",
            True,
            checks,
            problem,
            _reinstall_command(wheel) if wheel else None,
            links,
            _wheel_reason(p.driver_cuda, wheel),
        )
    # no NVIDIA GPU on this machine — the CUDA panel is shown greyed, not hidden
    headline = (
        "Apple GPU (MPS) detected" if p.mps else "No NVIDIA GPU detected — CPU only"
    )
    return Diagnosis(headline, "warn", False, checks, None, None, [], None)
