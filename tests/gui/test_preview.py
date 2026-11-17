"""Prompt-test / render-preview dialog logic + the thumbnail grid.

The success path needs a GPU model/data box; here we exercise the parts that run
anywhere — the grid, dialog construction, and validation."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")


class _FakeRecording:
    id = "recording_001"
    name = "Fish tank A"  # pickers show the name (§11); the id is the data


def test_prompt_text_extracts_cue():
    from detail.gui.preview import _prompt_text

    text_prompt = {
        "targets": [{"class_name": "m", "cue_type": "text", "cue": "rodent"}]
    }
    assert _prompt_text(text_prompt) == "rodent"
    assert _prompt_text({"targets": []}) == "?"


def _compact_frames(confidences_per_frame):
    """Compact prompt-test frames carrying the given per-prompt confidences.

    Built through the real ``_CompactFrame``, so the RLE round-trip the dialog depends on
    is exercised rather than stubbed.
    """
    from detail.core.dataclasses import FrameResult, PromptTestResult
    from detail.gui.preview import _CompactFrame

    out = []
    for i, per_prompt in enumerate(confidences_per_frame):
        results = {}
        for label, confidences in per_prompt.items():
            results[label] = FrameResult.from_detections(
                frame_idx=i,
                t_ns=i,
                detections=[_detection(c, box=k) for k, c in enumerate(confidences)],
            )
        out.append(
            _CompactFrame(
                PromptTestResult(
                    part_index=0,
                    frame_idx=i,
                    t_ns=i,
                    frame_image=np.zeros((16, 16, 3), np.uint8),
                    results=results,
                )
            )
        )
    return out


def _detection(confidence, box=0):
    from detail.core.dataclasses import Detection

    mask = np.zeros((16, 16), np.uint8)
    mask[box : box + 3, box : box + 3] = 1
    return Detection(
        mask=mask,
        bbox=(0.1, 0.1, 0.4, 0.4),
        class_name="mouse",
        confidence=confidence,
        object_id=None,
    )


def test_prompt_matrix_builds_a_cell_per_frame_and_prompt(qapp):
    from detail.gui.preview import _PromptMatrix

    frames = _compact_frames(
        [
            {"prompt_0": [0.9, 0.8, 0.2], "prompt_1": []},
            {"prompt_0": [0.7, 0.1], "prompt_1": [0.95]},
        ]
    )
    prompts = [("prompt_0", "mouse"), ("prompt_1", "rat")]
    colors = {"prompt_0": (230, 159, 0), "prompt_1": (86, 180, 233)}
    matrix = _PromptMatrix(frames, prompts, colors, 0.0)
    assert matrix.cell_count() == 4  # 2 frames × 2 prompts
    assert _PromptMatrix([], prompts, colors, 0.0).cell_count() == 0  # no results


def test_prompt_matrix_threshold_changes_the_counts_it_shows(qapp):
    """The whole point of running wide open: the same result re-counts as the threshold
    moves. A matrix that kept the run's own totals would show one answer forever."""
    from detail.gui.preview import _PromptMatrix

    frames = _compact_frames(
        [
            {"prompt_0": [0.9, 0.8, 0.2], "prompt_1": []},
            {"prompt_0": [0.7, 0.1], "prompt_1": [0.95]},
        ]
    )
    prompts = [("prompt_0", "mouse"), ("prompt_1", "rat")]
    colors = {"prompt_0": (230, 159, 0), "prompt_1": (86, 180, 233)}
    matrix = _PromptMatrix(frames, prompts, colors, 0.0)

    assert matrix.counts(0.0) == (6, 6)  # everything the model proposed
    assert matrix.counts(0.5) == (4, 6)  # 0.9, 0.8, 0.7, 0.95
    assert matrix.counts(0.92) == (1, 6)  # only 0.95
    assert matrix.counts(1.0) == (0, 6)

    # The header follows the slider, not the run.
    matrix.set_min_conf(0.5)
    assert "3 total" in matrix._headers["prompt_0"]._label.text()
    matrix.set_min_conf(0.75)
    assert "2 total" in matrix._headers["prompt_0"]._label.text()


def test_compact_frame_round_trips_masks_through_rle(qapp):
    """Masks are stored encoded so the dialog can hold every proposal; a decode that
    lost the mask would draw an empty overlay and look like "no detections"."""
    from detail.gui.preview import _CompactFrame  # noqa: F401  (built via helper)

    frame = _compact_frames([{"prompt_0": [0.9]}])[0]
    (restored,) = frame.detections("prompt_0", 0.0)
    assert restored.confidence == pytest.approx(
        0.9
    )  # FrameResult rows are float32 (§5)
    assert restored.mask.shape == (16, 16)
    assert restored.mask.sum() > 0  # the encoded mask survived the round trip


def test_prompt_test_dialog(qapp):
    from detail.core.dataclasses import ComputeTarget
    from detail.gui.preview import PromptTestDialog

    dlg = PromptTestDialog(
        [_FakeRecording()],
        model_id="sam3",
        prompt_text="mouse\nrodent",
        compute=ComputeTarget("cpu", None, None, None),
    )
    assert dlg._recording.count() == 1
    assert len(dlg._prompt_list()) == 2  # two candidate prompts
    assert dlg._validate() is None
    dlg._prompts.setPlainText("   \n  ")  # blank → invalid
    assert dlg._validate() is not None


def test_prompt_test_dialog_closed_set(qapp):
    from PyQt6.QtCore import Qt

    from detail.core.dataclasses import ComputeTarget
    from detail.gui.preview import PromptTestDialog

    dlg = PromptTestDialog(
        [_FakeRecording()],
        model_id="model_001",
        prompt_text="",
        compute=ComputeTarget("cpu", None, None, None),
        closed_set=True,
        classes=["mouse", "cage"],
    )
    assert dlg._recording.count() == 1
    assert dlg._class_list.count() == 2  # a checkable item per trained class
    assert dlg._checked_classes() == ["mouse", "cage"]  # all checked by default
    assert dlg._validate() is None  # no prompt required in closed-set mode
    for i in range(dlg._class_list.count()):  # uncheck all → nothing to preview
        dlg._class_list.item(i).setCheckState(Qt.CheckState.Unchecked)
    assert dlg._validate() is not None
