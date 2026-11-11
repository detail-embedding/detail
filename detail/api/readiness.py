"""Machine-global setup-readiness signals for the Home onboarding panel (§1).

Derived from the live checker framework (``api/checks``): ``readiness_signals`` runs the
checks **in-process** (cheap, no spawn) and maps them to the five booleans the guided layer
and the readiness panel read. The GUI recomputes from a **fresh-subprocess** ``run_checks``
and maps with ``signals_from``, so a mid-session install shows up on re-check without an app
restart. Qt-free, network-free, and never raises (a check that can't answer is not ``OK``).
"""

from dataclasses import dataclass

from detail.api.checks import run_checks
from detail.core.dataclasses import CheckResult, CheckStatus
from detail.models import imported as imported_store
from detail.storage import locations


@dataclass(frozen=True)
class ReadinessSignals:
    """The setup-checklist state the onboarding panel renders (§1).

    Each field is a cheap, network-free boolean:
        gpu: an NVIDIA/CUDA device is present — SAM3's hard prerequisite.
        hf_token: a Hugging Face token is configured — gates the gated downloads.
        sam3: SAM3 is usable — weights installed AND the sam3 backend package imports.
        sam3_backend: the sam3 backend package imports (the pip install half of SAM3).
        sam3_weights: the SAM3 weights are downloaded (the Hugging Face half of SAM3).
        detection_model: at least one usable detection model exists (SAM3 or an
            imported model), so the pipeline can run.
        has_experiment: at least one experiment exists in the managed workspace.
    """

    gpu: bool
    hf_token: bool
    sam3: bool
    sam3_backend: bool
    sam3_weights: bool
    detection_model: bool
    has_experiment: bool


def signals_from(results: list[CheckResult]) -> ReadinessSignals:
    """Map check results (plus two cheap workspace probes) to the five checklist booleans.

    ``sam3`` is honest now: weights alone is not enough — the ``sam3`` package must also
    import (``sam3_backend``), so "SAM 3 ready" can't be true while a run would crash at
    ``import sam3``."""
    by_id = {r.check_id: r for r in results}

    def ok(check_id: str) -> bool:
        result = by_id.get(check_id)
        return result is not None and result.status == CheckStatus.OK

    gpu = ok("cuda")
    sam3_backend = ok("sam3_backend")
    sam3_weights = ok("sam3_weights")
    sam3 = sam3_backend and sam3_weights
    # SAM3 only counts as a usable detector where it can actually run (CUDA); on any other
    # machine a usable model means an imported (CPU/MPS-capable) one.
    # A keypoint (pose) model in the catalog detects nothing (§12), so it does not count.
    detection_model = any(
        not imported_store.is_pose_record(rec) for rec in imported_store.list_records()
    ) or (sam3 and gpu)
    return ReadinessSignals(
        gpu=gpu,
        hf_token=ok("hf_token"),
        sam3=sam3,
        sam3_backend=sam3_backend,
        sam3_weights=sam3_weights,
        detection_model=detection_model,
        has_experiment=bool(locations.list_experiment_dirs()),
    )


def readiness_signals() -> ReadinessSignals:
    """The checklist state computed in-process (no subprocess spawn) — the cheap path for
    tests and non-GUI callers. The GUI recomputes from ``run_checks()`` (a fresh subprocess)
    then ``signals_from``, so a re-check reflects a mid-session install without restart.
    """
    return signals_from(run_checks(in_process=True))
