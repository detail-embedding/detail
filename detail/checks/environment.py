"""Import-sensitive environment checks (§1).

These probe the Python environment (is torch installed? does it see a GPU? does the
``sam3`` package import?), so their truth depends on what's importable *right now*. They
are marked ``isolation="subprocess"``: the runner executes them in a fresh interpreter, so
a mid-session ``pip install`` is reflected on the next re-check **without restarting the
app** — a C-extension like torch can't be swapped in a running process.

Backend imports stay inside ``run`` (Tenet 13): importing this module registers the checks
but pulls in no torch/sam3. ``find_spec`` locates a package without importing it.
"""

import os

from detail.core.dataclasses import CheckResult, CheckStatus
from detail.core.installed import is_importable
from detail.core.protocols import Check
from detail.core.registries import register_check
from detail.help import HelpTopic


def _has(module: str) -> bool:
    """Whether ``module`` is importable, without importing it (Tenet 13)."""
    return is_importable(module)


@register_check("torch")
class TorchInstalled(Check):
    check_id = "torch"
    label = "PyTorch"
    group = "Environment"
    isolation = "subprocess"
    fix = HelpTopic("getting-started", "gpu-and-cuda")

    def run(self, target=None) -> CheckResult:
        if _has("torch"):
            return CheckResult(self.check_id, CheckStatus.OK, "installed")
        return CheckResult(self.check_id, CheckStatus.FAIL, "torch is not installed")


@register_check("psutil")
class PsutilInstalled(Check):
    """`psutil` is a core dependency (#394): a missing one is a broken install, not a plain
    one, since it ships with every install path — worth a doctor complaint rather than a
    performance report that silently reads "unknown" for RAM."""

    check_id = "psutil"
    label = "psutil (RAM reporting)"
    group = "Environment"
    isolation = "subprocess"
    fix = HelpTopic("getting-started", "install")

    def run(self, target=None) -> CheckResult:
        if _has("psutil"):
            return CheckResult(self.check_id, CheckStatus.OK, "installed")
        return CheckResult(
            self.check_id,
            CheckStatus.FAIL,
            "psutil is not installed — the performance report's RAM field will read unknown",
        )


@register_check("cuda")
class CudaAvailable(Check):
    check_id = "cuda"
    label = "CUDA (GPU)"
    group = "Environment"
    isolation = "subprocess"
    fix = HelpTopic("getting-started", "gpu-and-cuda")

    def run(self, target=None) -> CheckResult:
        try:
            import torch  # a real import: only a CUDA-enabled build reports a device
        except ImportError:
            return CheckResult(self.check_id, CheckStatus.WARN, "torch not installed")
        if torch.cuda.is_available():
            return CheckResult(
                self.check_id, CheckStatus.OK, "a CUDA device is available"
            )
        return CheckResult(
            self.check_id,
            CheckStatus.WARN,
            "no CUDA device (SAM 3 needs one; CPU/MPS models still run)",
        )


@register_check("sam3_backend")
class Sam3Backend(Check):
    check_id = "sam3_backend"
    label = "SAM 3 backend package"
    group = "Environment"
    isolation = "subprocess"
    fix = HelpTopic("getting-started", "sam3-backend")

    def run(self, target=None) -> CheckResult:
        if _has("sam3"):
            return CheckResult(self.check_id, CheckStatus.OK, "importable")
        return CheckResult(
            self.check_id,
            CheckStatus.FAIL,
            "the sam3 package isn't installed (git clone + pip install)",
        )


@register_check("triton")
class TritonInstalled(Check):
    check_id = "triton"
    label = "Triton (Windows)"
    group = "Environment"
    isolation = "subprocess"
    fix = HelpTopic("getting-started", "sam3-backend")

    def run(self, target=None) -> CheckResult:
        if os.name != "nt":
            return CheckResult(
                self.check_id, CheckStatus.SKIPPED, "not needed off Windows"
            )
        if _has("triton"):
            return CheckResult(self.check_id, CheckStatus.OK, "installed")
        return CheckResult(
            self.check_id, CheckStatus.FAIL, "Triton is required for SAM 3 on Windows"
        )


@register_check("dinov3_backend")
class Dinov3Backend(Check):
    check_id = "dinov3_backend"
    label = "DINOv3 backend (transformers)"
    group = "Environment"
    isolation = "subprocess"
    fix = HelpTopic("getting-started", "install")

    def run(self, target=None) -> CheckResult:
        missing = [m for m in ("torch", "transformers") if not _has(m)]
        if missing:
            return CheckResult(
                self.check_id, CheckStatus.WARN, f"missing: {', '.join(missing)}"
            )
        return CheckResult(
            self.check_id, CheckStatus.OK, "torch + transformers installed"
        )
