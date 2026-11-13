"""Machine-global setup-readiness signals for the Home onboarding panel (§1).

Derived from the live checker framework (``api/checks``): ``readiness_signals`` runs the
checks **in-process** (cheap, no spawn) and maps them to the booleans the guided layer
and the readiness panel read. The GUI recomputes from a **fresh-subprocess** ``run_checks``
and maps with ``signals_from``, so a mid-session install shows up on re-check without an app
restart. Qt-free, network-free, and never raises (a check that can't answer is not ``OK``).
"""

from dataclasses import dataclass

from detail.api.checks import run_checks
from detail.core.dataclasses import CheckResult, CheckStatus
from detail.models import imported as imported_store
from detail.storage import guidance, locations
from detail.storage import models as models_store


@dataclass(frozen=True)
class ReadinessSignals:
    """The setup-checklist state the onboarding panel renders (§1).

    Each field is a cheap, network-free boolean.
        gpu: PyTorch can use a CUDA device — SAM3's hard prerequisite. Not "this machine
            has an NVIDIA GPU": a first install's CPU-only torch wheel reads ``False`` here
            on a box with one, which is what ``nvidia_gpu`` is for.
        nvidia_gpu: an NVIDIA GPU is present (its driver answers ``nvidia-smi``, or torch
            sees a CUDA device), whether or not PyTorch can use it yet. ``nvidia_gpu and
            not gpu`` is a fixable install, not a machine without a GPU.
        hf_token: a Hugging Face token is configured — gates the gated downloads.
        sam3: SAM3 is usable — weights installed AND the sam3 backend package imports.
        sam3_backend: the sam3 backend package imports (the pip install half of SAM3).
        sam3_weights: the SAM3 weights are downloaded (the Hugging Face half of SAM3).
        detection_model: at least one usable detection model exists (SAM3 or an
            imported model), so the pipeline can run.
        has_experiment: at least one experiment exists in the managed workspace.
        distilled_detector: a fast detector has been trained — one in any experiment's
            ``models/``, or a distilled one in the catalog. What keypoint setup finds the
            animal with (§26, #717).
        keypoint_student: a keypoint model has been distilled into the catalog, which is
            what keypoint setup and ``distill_keypoints`` both end with.
        keypoints_skipped: the person chose to go without a keypoint model
            (``storage/guidance.py``); the checklist then stops nudging, and says so.
        shap: SHAP is installed, so Cluster profile's default explainer runs rather than
            falling back to permutation importance. Optional.
    """

    gpu: bool
    nvidia_gpu: bool
    hf_token: bool
    sam3: bool
    sam3_backend: bool
    sam3_weights: bool
    detection_model: bool
    has_experiment: bool
    distilled_detector: bool
    keypoint_student: bool
    keypoints_skipped: bool
    shap: bool


def signals_from(results: list[CheckResult]) -> ReadinessSignals:
    """Map check results (plus cheap workspace and model-catalog probes) to the checklist.

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
        not imported_store.is_pose_record(rec)
        for rec in imported_store.list_records(role="detection")
    ) or (sam3 and gpu)
    experiments = locations.list_experiment_dirs()
    return ReadinessSignals(
        gpu=gpu,
        # A device torch can use is an NVIDIA one even where nvidia-smi isn't on PATH.
        nvidia_gpu=gpu or ok("nvidia_gpu"),
        hf_token=ok("hf_token"),
        sam3=sam3,
        sam3_backend=sam3_backend,
        sam3_weights=sam3_weights,
        detection_model=detection_model,
        has_experiment=bool(experiments),
        distilled_detector=_distilled("detection") or _trained_detector(experiments),
        keypoint_student=_distilled("keypoints") or _trained_student(experiments),
        # Never raises: a damaged guidance.yaml reads as not opted out (this module's promise).
        keypoints_skipped=guidance.keypoints_opted_out(),
        shap=ok("shap"),
    )


def _distilled(role: str) -> bool:
    """A model of ``role`` that this machine distilled sits in the catalog."""
    return any(
        record.get("source") == "distilled"
        for record in imported_store.list_records(role=role)
    )


def _trained_student(experiment_dirs) -> bool:
    """Any experiment holds a finished keypoint student (§12 *Distill keypoints*): on main's
    classes a distill registers its student in the experiment (``task: pose``), where the
    stage before §26 promoted it to the catalog, so both places count."""
    for experiment_dir in experiment_dirs:
        for model_id in models_store.discover_models(experiment_dir):
            try:
                record = models_store.read_model_yaml(
                    models_store.model_dir(experiment_dir, model_id)
                )
            except (FileNotFoundError, ValueError):
                continue
            if record.get("status") == "complete" and record.get("task") == "pose":
                return True
    return False


def _trained_detector(experiment_dirs) -> bool:
    """Any experiment holds a finished detector it trained (a keypoint student is a trained
    model too, and detects nothing)."""
    for experiment_dir in experiment_dirs:
        for model_id in models_store.discover_models(experiment_dir):
            try:
                record = models_store.read_model_yaml(
                    models_store.model_dir(experiment_dir, model_id)
                )
            except (FileNotFoundError, ValueError):
                continue
            if (
                record.get("status") == "complete"
                and record.get("role") != "keypoints"
                and record.get("task") != "pose"
            ):
                return True
    return False


def set_keypoints_skipped(skipped: bool) -> None:
    """Go without a keypoint model on this machine (``True``), or take that back (§26, #717).

    The setup checklist then stops nudging for keypoint setup and says what was given up:
    alignment and pose in the embedding, and everything that reads them.
    """
    guidance.set_keypoints_skipped(skipped)


def readiness_signals() -> ReadinessSignals:
    """The checklist state computed in-process (no subprocess spawn) — the cheap path for
    tests and non-GUI callers. The GUI recomputes from ``run_checks()`` (a fresh subprocess)
    then ``signals_from``, so a re-check reflects a mid-session install without restart.
    """
    return signals_from(run_checks(in_process=True))
