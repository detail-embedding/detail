"""A local-only preview must not be handed the run's compute target as a device.

**The class.** A synchronous preview (prompt test, keypoints test) blocks its caller and
never goes through an executor, so it can only run on this machine. The forms that open
one can target a registered cluster, and they hand the preview the *run's* target. A
``backend="remote"`` target names where a full run is submitted, not a device; read as a
device it failed every check written for one. The label, distill and detection forms'
**Test prompts…** reached ``Recording.run_prompt_test`` with it, which compared
``"remote"`` against the model's ``supported_backends`` and refused:
``backend 'remote' not supported by 'sam3'`` — a preview that could not run at all from
any form set to Tillicum.

**How it hid.** Every prompt-test test built a local ``ComputeTarget``; the keypoints test
had already met the same shape and resolved it inline, so the fix existed one module away.

**The rule.** A preview resolves its target through ``core.utils.preview_compute``: a
remote (or absent) target becomes this machine's best supported backend, a local one is
kept. The dialog says so when the form is remote
(``tests/gui/test_preview.py::test_prompt_test_dialog_says_a_remote_form_previews_locally``);
the keypoints teacher test's own case is in ``tests/pipeline/test_keypoint_preview.py``. The CLI's
``detection prompt-test`` offers no ``remote`` backend, so it cannot reach this.
"""

import json

import numpy as np

from detail.core.dataclasses import ComputeTarget
from detail.core.dataclasses import Detection
from detail.core.dataclasses import FrameResult
from detail.core.registries import MODELS
from detail.core.registries import register_model
from detail.core.utils import preview_compute
from tests.api.test_recording import _FakeDetector
from tests.api.test_recording import _add_video
from tests.api.test_recording import _make_recording
from tests.api.test_recording import _text_prompt


class _PredictingFakeDetector(_FakeDetector):
    def predict(self, frame, prompt):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        return FrameResult.from_detections(
            0, 0, [Detection(mask, (0.25, 0.25, 0.75, 0.75), "mouse", 0.9, None)]
        )

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]


def test_preview_compute_keeps_a_local_target_and_resolves_a_remote_one():
    local = ComputeTarget("cpu", None, None, None)
    assert preview_compute(local, ["cpu", "cuda"]) is local
    remote = ComputeTarget("remote", None, None, None, "tillicum")
    assert preview_compute(remote, ["cpu"]).backend == "cpu"
    assert preview_compute(None, ["cpu"]).backend == "cpu"


def test_run_prompt_test_remote_target_runs_on_this_machine(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    remote = ComputeTarget("remote", None, None, None, "tillicum")
    register_model("fake_detector")(_PredictingFakeDetector)
    try:
        results = recording.run_prompt_test(
            "fake_detector",
            [_text_prompt()],
            n_sample_frames=2,
            compute=remote,
            save_as="from_remote_form",
        )
    finally:
        MODELS.pop("fake_detector", None)
    assert results
    saved = recording.dir / "detections" / "prompt_tests" / "from_remote_form"
    record = json.loads((saved / "test.json").read_text(encoding="utf-8-sig"))
    assert record["compute"]["backend"] == "cpu"  # the fake's only backend
