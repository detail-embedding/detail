"""A run of the stage before §26 opens on its run pages and draws its points (§26 catch-up).

The pages look a run's viewer up by its module; ``keypoints`` is no longer a module, so the
pages fall back to ``keypoint_run_preview.legacy_preview_widget`` — the API half is
``tests/api/test_legacy_keypoint_runs.py``, whose fixture writes the old run as the old stage
did.
"""

import pytest

from tests import experiments
from tests.api.test_legacy_keypoint_runs import (  # noqa: F401  (fixtures)
    legacy,
    legacy_aligned,
)
from tests.gui.conftest import drain_until


@pytest.fixture
def legacy_with_video(legacy):
    experiments.write_video(legacy.recording.dir / "video" / "part_001.mp4")
    return legacy


def test_the_run_pages_open_an_old_run_on_its_points(qapp, legacy_with_video):
    from detail.gui.keypoint_run_preview import LegacyKeypointsRunPreview
    from detail.gui.pages import RecordingRunDetailPage, RunDetailPage

    proj, recording = legacy_with_video.experiment, legacy_with_video.recording
    for page in (
        RunDetailPage(proj, "run_kp"),
        RecordingRunDetailPage(proj, recording.id, "keypoints", "kp_01"),
    ):
        (view,) = page.findChildren(LegacyKeypointsRunPreview)
        assert drain_until(qapp, lambda: hasattr(view, "_scrubber")), "never loaded"
        summary = view._summary.text()
        assert "dinov3_match · 2 points · not visible below 0.4" in summary
        kind, with_points = view._compute(0)  # frame 0: the old run placed both points
        assert kind == "img"
        _actual, rgb, detections = view._preview.frame(0, 0, with_masks=True)
        assert (with_points != view._overlay(rgb, detections)).any()  # the points


def test_an_old_aligned_embedding_run_opens_on_its_scatter(qapp, legacy_aligned):
    from detail.gui.pages import RunDetailPage
    from detail.gui.run_preview import EmbeddingRunPreview

    page = RunDetailPage(legacy_aligned.experiment, "run_emb")
    (view,) = page.findChildren(EmbeddingRunPreview)
    assert drain_until(qapp, lambda: getattr(view, "_preview", None) is not None)
    assert view._status.text() == "Click a point to inspect"
