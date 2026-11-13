"""Shared setup-step model for the readiness panel, banners, and sidebar marker (§1).

One ordering + done-predicate over ``ReadinessSignals`` so the three guidance surfaces —
the Home panel, the Configuration-page banners, and the sidebar next-step marker — never
drift (the design's "both surfaces read the same signals"). Qt-free.
"""

from dataclasses import dataclass

from detail.api.readiness import ReadinessSignals

#: The keypoint setup step's kind: its Open button, its banner, and its opt-out.
KEYPOINT_STEP = "keypoint_setup"

#: The optional Cluster profile step's kind: SHAP.
PROFILE_DESCRIBE_STEP = "profile_describe"

#: What going without a keypoint model gives up, said where the choice is made (§26, #717).
KEYPOINTS_SKIPPED_COSTS = (
    "aligning crops to the animal's body axis in the embedding",
    "pose in the embedding, and so pose-aware clustering and query",
    "keypoint exports and the keypoints video",
    "the keypoint audit",
)


@dataclass(frozen=True)
class Step:
    """One machine-setup step.

    ``kind`` is the sidebar nav kind the panel/marker point at (``None`` for a
    not-yet-built step, e.g. the Quick-segment wow). ``guides`` marks the steps that
    carry the active nudge — the sidebar ``●`` and a page banner: only the gated,
    actionable steps (Hugging Face, install a detection model, and GPU on a machine with
    an NVIDIA GPU PyTorch can't use yet). GPU with no NVIDIA present is terminal, and it,
    the coming-soon wow, and the optional distill step never guide.
    """

    kind: str | None
    label: str
    done: bool
    guides: bool


def readiness_steps(s: ReadinessSignals) -> list[Step]:
    """The ordered machine-setup checklist for the current signals."""
    steps = [
        # An NVIDIA GPU that PyTorch can't use is a fixable install (the GPU / CUDA page
        # names the fix), so it guides; no NVIDIA GPU at all is nothing to act on.
        Step("gpu_cuda", gpu_label(s), s.gpu, guides=s.nvidia_gpu and not s.gpu),
        # Hugging Face is needed on every machine: DINOv3 (which runs on CPU/MPS) is
        # gated too, not just the CUDA-only SAM3.
        Step("hugging_face", "Hugging Face access", s.hf_token, guides=True),
    ]
    if s.nvidia_gpu:
        # SAM 3 runs here (once PyTorch can use the GPU, which the GPU step guides to), so
        # it is the guiding "get a detector" step — split into its two honest halves: the
        # pip backend (its own Configuration page shows the install commands) and the
        # Hugging Face weights (downloaded from the Model manager).
        steps.append(
            Step("sam3_backend", "SAM 3 backend package", s.sam3_backend, guides=True)
        )
        steps.append(
            Step("global_models", "SAM 3 weights", s.sam3_weights, guides=True)
        )
    else:
        # SAM 3 is CUDA-only. Don't hide it on a machine where it can't run (grey-out,
        # don't hide): show both install halves as non-guiding steps, and guide the user
        # to a runnable (imported / distilled) detector instead.
        steps.append(
            Step(
                "sam3_backend",
                "SAM 3 backend package (needs an NVIDIA GPU)",
                s.sam3_backend,
                guides=False,
            )
        )
        steps.append(
            Step("global_models", "SAM 3 weights", s.sam3_weights, guides=False)
        )
        steps.append(
            Step(
                "global_models",
                "Install a runnable detection model",
                s.detection_model,
                guides=True,
            )
        )
    steps.append(Step("quick_segment", "See it segment a clip", False, guides=False))
    steps.append(
        Step(
            "distill_model",
            "Recommended: distill a fast model",
            s.distilled_detector,
            guides=False,
        )
    )
    # Every experiment is set up with a keypoint model unless the person opts out (§26,
    # #717): alignment and pose in the embedding need one. It nudges once it can be done —
    # keypoint setup finds the animal with a distilled detector.
    steps.append(
        Step(
            KEYPOINT_STEP,
            (
                "Keypoints skipped: Analysis cannot align or use pose"
                if s.keypoints_skipped
                else "Set up keypoints for Analysis"
            ),
            s.keypoint_student or s.keypoints_skipped,
            guides=s.distilled_detector,
        )
    )
    # Optional: it never guides (no banner, no nudge), so there is nothing to skip — going
    # without it costs a fallback, which its own page and the profile window both name.
    steps.append(
        Step(
            PROFILE_DESCRIBE_STEP,
            profile_describe_label(s),
            s.shap,
            guides=False,
        )
    )
    return steps


def gpu_label(s: ReadinessSignals) -> str:
    """The GPU step's label: ready, an NVIDIA GPU PyTorch can't use yet, or none at all."""
    if s.gpu:
        return "GPU / CUDA"
    if s.nvidia_gpu:
        return "GPU / CUDA (PyTorch can't use the GPU yet)"
    return "GPU / CUDA (no NVIDIA GPU detected)"


def profile_describe_label(s: ReadinessSignals) -> str:
    """The optional step's label: done, or what is still missing."""
    if s.shap:
        return "Cluster profile"
    return "Optional: Cluster profile (needs SHAP)"


def next_step_kind(s: ReadinessSignals) -> str | None:
    """The kind of the next step still to do — the sidebar ``●`` target and the banner
    trigger. Walks the guiding steps (GPU on an NVIDIA box PyTorch can't use yet →
    Hugging Face → install), skipping the coming-soon wow and the optional distill step.
    ``None`` once setup is done."""
    for step in readiness_steps(s):
        if step.guides and not step.done:
            return step.kind
    return None


def is_setup_complete(s: ReadinessSignals) -> bool:
    """Whether every required (guiding) step is done."""
    return next_step_kind(s) is None


def guides_step(kind: str, s: ReadinessSignals) -> Step | None:
    """The step a Configuration page should banner, when that page's step is currently a
    guiding, incomplete one; else ``None`` (no banner)."""
    for step in readiness_steps(s):
        if step.kind == kind and step.guides and not step.done:
            return step
    return None
