"""Shared setup-step model for the readiness panel, banners, and sidebar marker (§1).

One ordering + done-predicate over ``ReadinessSignals`` so the three guidance surfaces —
the Home panel, the Configuration-page banners, and the sidebar next-step marker — never
drift (the design's "both surfaces read the same signals"). Qt-free.
"""

from dataclasses import dataclass

from detail.api.readiness import ReadinessSignals


@dataclass(frozen=True)
class Step:
    """One machine-setup step.

    ``kind`` is the sidebar nav kind the panel/marker point at (``None`` for a
    not-yet-built step). ``done`` is read from a signal, never written as a literal: a step
    that can never be ticked tells the user they have not done something they did
    (``requirements.a_checklist_step_reads_its_state``). ``guides`` marks the steps that
    carry the active nudge — the sidebar ``●`` and a page banner: only the gated,
    actionable steps (Hugging Face, install a detection model). GPU (terminal when no
    NVIDIA is present), Quick segment, and the optional distill step never guide.
    """

    kind: str | None
    label: str
    done: bool
    guides: bool


def readiness_steps(s: ReadinessSignals) -> list[Step]:
    """The ordered machine-setup checklist for the current signals."""
    steps = [
        Step(
            "gpu_cuda",
            "GPU / CUDA" if s.gpu else "GPU / CUDA (no NVIDIA GPU detected)",
            s.gpu,
            guides=False,
        ),
        # Hugging Face is needed on every machine: DINOv3 (which runs on CPU/MPS) is
        # gated too, not just the CUDA-only SAM3.
        Step("hugging_face", "Hugging Face access", s.hf_token, guides=True),
    ]
    if s.gpu:
        # SAM 3 runs here, so it is the guiding "get a detector" step — split into its two
        # honest halves: the pip backend (its own Configuration page shows the install
        # commands) and the Hugging Face weights (downloaded from the Model manager).
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
    # Quick segment runs SAM 3 on a GPU box. Without one, its model list offers only imported
    # and distilled detectors, so the step says what it needs, as the SAM 3 rows do.
    steps.append(
        Step(
            "quick_segment",
            (
                "See it segment a clip"
                if s.gpu
                else "See it segment a clip (needs an NVIDIA GPU or an installed "
                "detection model)"
            ),
            s.quick_segment_run,
            guides=False,
        )
    )
    steps.append(
        Step(
            "distill_model",
            "Recommended: distill a fast model",
            s.distilled_model,
            guides=False,
        )
    )
    return steps


def next_step_kind(s: ReadinessSignals) -> str | None:
    """The kind of the next step still to do — the sidebar ``●`` target and the banner
    trigger. Walks the guiding steps (Hugging Face → install), skipping GPU, the
    Quick segment step, and the optional distill step. ``None`` once setup is done."""
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
