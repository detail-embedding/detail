"""Shared setup-step model (readiness_guide) — one source of truth for the guidance."""

import itertools

from detail.api.readiness import ReadinessSignals
from detail.gui import readiness_guide


def _s(**kw):
    # `sam3` (the combined signal other consumers read) defaults to the AND of its two
    # halves unless a test overrides it.
    kw.setdefault(
        "sam3", kw.get("sam3_backend", False) and kw.get("sam3_weights", False)
    )
    base = dict(
        gpu=False,
        hf_token=False,
        sam3=False,
        sam3_backend=False,
        sam3_weights=False,
        detection_model=False,
        has_experiment=False,
    )
    base.update(kw)
    return ReadinessSignals(**base)


def _kinds(steps):
    return [step.kind for step in steps]


def test_hugging_face_guides_on_every_machine():
    # DINOv3 is gated too and runs on CPU/MPS, so HF access is needed with or without
    # an NVIDIA GPU.
    def hf(sig):
        return next(
            s for s in readiness_guide.readiness_steps(sig) if s.kind == "hugging_face"
        )

    assert hf(_s(gpu=True)).guides
    assert hf(_s(gpu=False)).guides


def test_sam3_splits_into_backend_and_weights_on_gpu():
    # On a GPU box SAM 3 is the guiding "get a detector" step, split into its two honest
    # halves: the pip backend (its own Configuration page) and the HF weights (Model manager).
    steps = readiness_guide.readiness_steps(
        _s(gpu=True, sam3_backend=True, sam3_weights=True)
    )
    backend = next(s for s in steps if s.kind == "sam3_backend")
    assert backend.label == "SAM 3 backend package"
    assert backend.done is True and backend.guides is True
    weights = next(s for s in steps if s.kind == "global_models")
    assert weights.label == "SAM 3 weights"
    assert weights.done is True and weights.guides is True


def test_sam3_is_shown_greyed_off_gpu_not_hidden():
    # grey-out, don't hide: on a non-CUDA machine both SAM 3 halves still appear (with their
    # status) as non-guiding steps, and a separate step guides to a runnable detector.
    steps = readiness_guide.readiness_steps(
        _s(gpu=False, sam3_backend=True, sam3_weights=True, detection_model=True)
    )
    backend = next(s for s in steps if s.kind == "sam3_backend")
    assert backend.done is True and backend.guides is False
    assert "needs an NVIDIA GPU" in backend.label
    weights = next(s for s in steps if s.label == "SAM 3 weights")
    assert weights.done is True and weights.guides is False
    runnable = next(s for s in steps if s.label == "Install a runnable detection model")
    assert runnable.done is True and runnable.guides is True


def test_guiding_steps_on_gpu_are_hf_and_both_sam3_halves():
    steps = readiness_guide.readiness_steps(_s(gpu=True))
    guiding = {s.kind for s in steps if s.guides}
    assert guiding == {"hugging_face", "sam3_backend", "global_models"}


def test_next_step_walks_hf_then_sam3_backend_then_weights():
    assert readiness_guide.next_step_kind(_s(gpu=True)) == "hugging_face"
    assert readiness_guide.next_step_kind(_s(gpu=True, hf_token=True)) == "sam3_backend"
    assert (
        readiness_guide.next_step_kind(_s(gpu=True, hf_token=True, sam3_backend=True))
        == "global_models"
    )
    assert (
        readiness_guide.next_step_kind(
            _s(gpu=True, hf_token=True, sam3_backend=True, sam3_weights=True)
        )
        is None
    )


def test_next_step_no_gpu_path():
    # no NVIDIA: GPU is terminal (never the target); HF is still needed (DINO), then a
    # detection model completes setup. The SAM 3 halves are non-guiding here.
    assert readiness_guide.next_step_kind(_s(gpu=False)) == "hugging_face"
    assert (
        readiness_guide.next_step_kind(_s(gpu=False, hf_token=True)) == "global_models"
    )
    assert (
        readiness_guide.next_step_kind(
            _s(gpu=False, hf_token=True, detection_model=True)
        )
        is None
    )


def test_is_setup_complete_requires_hf_and_the_right_detector():
    # complete = the guiding steps done: Hugging Face access + a detector. On an NVIDIA box
    # the detector is SAM 3 (both halves); off-GPU it's any detection model. GPU never gates.
    for gpu, hf, backend, weights, det in itertools.product([False, True], repeat=5):
        s = _s(
            gpu=gpu,
            hf_token=hf,
            sam3_backend=backend,
            sam3_weights=weights,
            detection_model=det,
        )
        expected = hf and ((backend and weights) if gpu else det)
        assert readiness_guide.is_setup_complete(s) is bool(expected)


def test_guides_step_only_for_incomplete_guiding_page():
    s = _s(gpu=True)  # hf + both sam3 halves incomplete
    assert readiness_guide.guides_step("hugging_face", s) is not None
    assert readiness_guide.guides_step("sam3_backend", s) is not None
    assert readiness_guide.guides_step("gpu_cuda", s) is None  # gpu never guides
    hf_done = _s(gpu=True, hf_token=True)
    assert (
        readiness_guide.guides_step("hugging_face", hf_done) is None
    )  # done → no banner
