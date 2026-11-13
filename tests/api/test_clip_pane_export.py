"""An enlarged clip-window pane written to a file (``api.clip_pane_export``).

Pinned: the file holds **the pane** — the masks on white, or the keypoints on white — at the
footage's own size, drawn with the settings it was handed; it covers the frames asked for at the
rate asked for; and a run with no keypoints says so rather than writing a white film.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from detail.api import clip_pane_export as pane_export
from detail.api.run_preview import ClipQueryPreview
from tests.api.test_clip_views import H, W, _walking, count_decodes


def _export(recording, path, pane, settings=None, **kwargs):
    return pane_export.export_clip_pane(
        recording.dir.parent.parent,
        recording_id=recording.id,
        embed_run_id="emb_01",
        track_id=0,
        first=0,
        last=4,
        pane=pane,
        settings=settings or pane_export.ClipViewSettings(),
        path=path,
        **kwargs,
    )


def test_the_masks_pane_is_written_at_full_size_every_frame(tmp_path):
    recording = _walking(tmp_path)
    path = _export(recording, tmp_path / "masks.mp4", pane_export.MASKS)
    cap = cv2.VideoCapture(str(path))
    try:
        assert round(cap.get(cv2.CAP_PROP_FPS)) == 30, "the recording's own rate"
        ok, first = cap.read()
        count = 1
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    assert ok and first.shape[:2] == (H, W), "the pane at the footage's size"
    assert count == 5
    # On white, with the subject in it: most of the picture is near-white, some is not.
    near_white = (first > 230).all(axis=2).mean()
    assert 0.5 < near_white < 1.0


def test_a_gif_samples_the_rate_asked_and_follows_the_settings(tmp_path):
    recording = _walking(tmp_path)
    plain = _export(recording, tmp_path / "plain.gif", pane_export.MASKS, fps=15)
    trailed = _export(
        recording,
        tmp_path / "trail.gif",
        pane_export.MASKS,
        settings=pane_export.ClipViewSettings(mask_trail=True),
        fps=15,
    )
    with Image.open(plain) as a, Image.open(trailed) as b:
        assert a.n_frames == 3  # 0, 67 and 133 ms of a 133 ms clip
        assert a.info["duration"] == 70  # 66.7 ms, stored to the hundredth
        assert a.size == (W, H)
        a.seek(2)
        b.seek(2)
        before = np.asarray(a.convert("RGB")).astype(int)
        after = np.asarray(b.convert("RGB")).astype(int)
    assert np.abs(after - before).sum() > 0, "Mask trail on draws the onion skin"


def test_the_keypoints_pane_is_the_pose_on_white(tmp_path):
    recording = _walking(tmp_path)
    path = _export(recording, tmp_path / "pose.gif", pane_export.KEYPOINTS)
    views = ClipQueryPreview(recording.dir.parent.parent).clip_views(
        recording.id, "emb_01", 0, 0, masks_on_frame=False, points_on_frame=False
    )
    with Image.open(path) as gif:
        drawn = np.asarray(gif.convert("RGB")).astype(int)
    assert (
        np.abs(drawn - views.pose.astype(int)).mean() < 8
    ), "the pose pane, not the masks"


def test_a_run_without_keypoints_says_so(tmp_path):
    recording = _walking(tmp_path, keypoints=False)
    with pytest.raises(ValueError, match="no keypoints"):
        _export(recording, tmp_path / "pose.gif", pane_export.KEYPOINTS)
    assert not (tmp_path / "pose.gif").exists()


def test_an_unknown_pane_is_refused(tmp_path):
    with pytest.raises(ValueError, match="pane must be one of"):
        pane_export.export_clip_pane(
            tmp_path, recording_id="r", embed_run_id="e", track_id=0, first=0, last=1,
            pane="footage", settings=pane_export.ClipViewSettings(),
            path=tmp_path / "x.gif",
        )  # fmt: skip


def test_the_frames_are_every_one_or_one_per_period():
    t_ns = np.arange(10, dtype=np.int64) * int(1e9 / 30)
    assert pane_export.pane_frames(t_ns, 2, 6, None) == [2, 3, 4, 5, 6]
    assert pane_export.pane_frames(t_ns, 0, 9, 10) == [0, 3, 6, 9]


def test_a_pane_gif_at_half_the_rate_takes_every_other_frame():
    """Through ``animation.frame_at_offset``: with rounded frame times an exact lookup took the
    frame before every other one it meant (``test_animation.py`` has the arithmetic)."""
    from tests.api.test_animation import rounded_frame_times

    t_ns = rounded_frame_times(12)
    assert pane_export.pane_frames(t_ns, 1, 9, 15) == [1, 3, 5, 7, 9]


@pytest.mark.parametrize("pane", pane_export.PANES)
def test_a_pane_export_decodes_no_video(tmp_path, monkeypatch, pane):
    """The file holds a pane — masks or keypoints on white, both stored data — so writing it
    opens no video, whatever the clip window's footage ticks are."""
    recording = _walking(tmp_path)
    settings = pane_export.ClipViewSettings(
        masks_on_frame=True, points_on_frame=True, trail=True, mask_trail=True
    )
    calls = count_decodes(monkeypatch)
    _export(recording, tmp_path / "pane.mp4", pane, settings)
    assert calls == []
