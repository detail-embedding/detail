"""Shared setup-step model (readiness_guide) — one source of truth for the guidance.

**The class: "PyTorch can't use a CUDA device" read as "this machine has no NVIDIA GPU".**
``torch.cuda.is_available()`` (the ``cuda`` check, ``ReadinessSignals.gpu``) is ``False`` on a
box with a working NVIDIA driver whenever the torch build can't use it, and a first install
gets the CPU-only wheel. The checklist said *no NVIDIA GPU detected*, the SAM 3 rows said they
*need an NVIDIA GPU*, and an extra "runnable detection model" row steered a new user away from
the GPU they have, while the GPU / CUDA page (which reads ``nvidia-smi``) said the opposite.
Nothing failed: every label was a plausible string. The presence question is answered by the
driver (the ``nvidia_gpu`` check, ``ReadinessSignals.nvidia_gpu``); ``gpu`` answers only whether
torch can use it. ``test_an_nvidia_gpu_pytorch_cannot_use_is_not_called_no_gpu`` pins the
checklist; the SAM 3 backend page's note is pinned in ``test_configuration_panel.py``
(``test_sam3_backend_page_says_pytorch_cannot_use_the_gpu_rather_than_none``).

The presence probe hid an instance of its own: 600-series drivers print ``KMD Version:`` /
``CUDA UMD Version:`` in ``nvidia-smi``'s header, and a parser reading only ``Driver Version:``
called a working GPU absent (the CUDA doctor said so on a 616.92 driver). Both header forms
are pinned in ``tests/core/test_hostinfo.py::test_nvidia_smi_reads_the_driver_and_its_cuda``.
"""

import itertools

import pytest

from detail.api.readiness import ReadinessSignals
from detail.gui import readiness_guide


def _s(**kw):
    # `sam3` (the combined signal other consumers read) defaults to the AND of its two
    # halves unless a test overrides it.
    kw.setdefault(
        "sam3", kw.get("sam3_backend", False) and kw.get("sam3_weights", False)
    )
    # A GPU torch can use is an NVIDIA one; a test of the CPU-only-torch case sets
    # `nvidia_gpu=True, gpu=False` itself.
    kw.setdefault("nvidia_gpu", kw.get("gpu", False))
    base = dict(
        gpu=False,
        nvidia_gpu=False,
        hf_token=False,
        sam3=False,
        sam3_backend=False,
        sam3_weights=False,
        detection_model=False,
        has_experiment=False,
        distilled_detector=False,
        keypoint_student=False,
        keypoints_skipped=False,
        shap=False,
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


def test_an_nvidia_gpu_pytorch_cannot_use_is_not_called_no_gpu():
    # A first install on an NVIDIA box gets a CPU-only torch wheel: `cuda` says no device
    # while the driver is right there. The checklist read that as "no NVIDIA GPU detected",
    # contradicting the GPU / CUDA page, and steered to the no-GPU detector path.
    steps = readiness_guide.readiness_steps(_s(gpu=False, nvidia_gpu=True))
    labels = [s.label for s in steps]
    gpu = steps[0]
    assert gpu.kind == "gpu_cuda"
    assert gpu.label == "GPU / CUDA (PyTorch can't use the GPU yet)"
    assert gpu.done is False and gpu.guides is True, "a fixable install, not terminal"
    assert not any("no NVIDIA GPU" in label for label in labels)
    assert not any("needs an NVIDIA GPU" in label for label in labels)
    assert "Install a runnable detection model" not in labels
    # SAM 3 runs here once PyTorch is fixed, so it guides as on any NVIDIA box.
    backend = next(s for s in steps if s.kind == "sam3_backend")
    assert backend.label == "SAM 3 backend package" and backend.guides is True
    # The fix comes first: it is the step the nudge points at.
    assert readiness_guide.next_step_kind(_s(gpu=False, nvidia_gpu=True)) == "gpu_cuda"
    assert not readiness_guide.is_setup_complete(
        _s(
            gpu=False,
            nvidia_gpu=True,
            hf_token=True,
            sam3_backend=True,
            sam3_weights=True,
            detection_model=True,
        )
    )


@pytest.mark.parametrize(
    "gpu, nvidia_gpu, label",
    [
        (True, True, "GPU / CUDA"),
        (False, True, "GPU / CUDA (PyTorch can't use the GPU yet)"),
        (False, False, "GPU / CUDA (no NVIDIA GPU detected)"),
    ],
)
def test_gpu_label_names_which_gap_it_is(gpu, nvidia_gpu, label):
    step = readiness_guide.readiness_steps(_s(gpu=gpu, nvidia_gpu=nvidia_gpu))[0]
    assert step.label == label
    # Only the fixable gap nudges: no NVIDIA GPU at all is nothing to act on.
    assert step.guides is (nvidia_gpu and not gpu)


def test_is_setup_complete_requires_hf_and_the_right_detector():
    # complete = the guiding steps done: Hugging Face access + a detector. On an NVIDIA box
    # the detector is SAM 3 (both halves); off-GPU it's any detection model. GPU gates only
    # where PyTorch can't use an NVIDIA GPU that is there (the test above).
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


# --- keypoint setup (§26, #717) -------------------------------------------------------- #
_READY = dict(gpu=True, hf_token=True, sam3_backend=True, sam3_weights=True)


def _keypoint_step(s):
    (step,) = [
        x
        for x in readiness_guide.readiness_steps(s)
        if x.kind == readiness_guide.KEYPOINT_STEP
    ]
    return step


def test_keypoint_setup_nudges_once_a_detector_can_find_the_animal():
    # Before a distilled detector it cannot be done, so it does not nudge.
    assert not _keypoint_step(_s(**_READY)).guides
    assert readiness_guide.is_setup_complete(_s(**_READY))
    with_detector = _s(**_READY, distilled_detector=True)
    assert _keypoint_step(with_detector).guides
    assert (
        readiness_guide.next_step_kind(with_detector) == readiness_guide.KEYPOINT_STEP
    )
    assert not readiness_guide.is_setup_complete(with_detector)


def test_keypoint_setup_is_done_by_a_student_or_by_opting_out():
    student = _s(**_READY, distilled_detector=True, keypoint_student=True)
    assert _keypoint_step(student).done
    assert readiness_guide.is_setup_complete(student)
    skipped = _s(**_READY, distilled_detector=True, keypoints_skipped=True)
    step = _keypoint_step(skipped)
    assert step.done and "cannot align or use pose" in step.label
    assert readiness_guide.is_setup_complete(skipped)


def test_the_distill_step_is_done_by_a_trained_detector():
    (distill,) = [
        x
        for x in readiness_guide.readiness_steps(_s(distilled_detector=True))
        if x.kind == "distill_model"
    ]
    assert distill.done


def test_the_optional_cluster_profile_step_needs_only_shap():
    """The optional step is SHAP alone: done once SHAP is in, never a banner either way."""

    def step(sig):
        (found,) = [
            s
            for s in readiness_guide.readiness_steps(sig)
            if s.kind == readiness_guide.PROFILE_DESCRIBE_STEP
        ]
        return found

    missing = step(_s())
    assert not missing.done and not missing.guides
    assert missing.label == "Optional: Cluster profile (needs SHAP)"
    ready = step(_s(shap=True))
    assert ready.done and ready.label == "Cluster profile"
