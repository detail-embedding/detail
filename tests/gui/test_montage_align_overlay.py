"""**Export ▾ → Export overlay…** on a clip grid (§22 *Align to heading, and Overlay*).

remote-detail's montage window had an *Overlay* tick that replaced its grid with the clips drawn
over each other, and **Export overlay…** to write it. On main's grid (``live_montage`` +
``montage_export``, every page's) the orientation is **Show ▾**'s *Aligned* and the overlay is
an export: it writes the picks while picking, else the grid's first
:data:`~detail.api.clip_montage.OVERLAY_CLIPS`, a PNG when every clip is one frame and an MP4 (in
the codec asked) or GIF otherwise, and is greyed with the reason while the run cannot turn its
clips to a heading.

Built on ``tests/api/test_montage_align._turning`` — one animal turning on the spot, with
keypoints and a heading definition — so every file asserted on was drawn by the montage's own
renderer. The turn itself is pinned in the API test.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402

from detail.api import clip_montage as cm  # noqa: E402
from tests.api.test_montage_align import FRAME_NS, _turning  # noqa: E402
from tests.gui.conftest import drain_until  # noqa: E402

ORIGIN = cm.MontageOrigin(run_id="an_07", node=7, name=None)
NO_MODIFIER = Qt.KeyboardModifier.NoModifier


def _grid(qapp, tmp_path, *, spans=None, aligned_run=True):
    from detail.gui import live_montage
    from detail.gui.montage_export import ExportSource

    recording = _turning(tmp_path, aligned_run=aligned_run)
    spans = spans or [(0, 4), (1, 3), (3, 4), (0, 1)]
    clips = [
        cm.MontageClip(
            i + 1, recording.id, "larva", 0, "emb_01", a * FRAME_NS, b * FRAME_NS
        )
        for i, (a, b) in enumerate(spans)
    ]
    experiment_dir = recording.dir.parent.parent
    montage = live_montage.LiveMontage(
        aligned_key="interpret/test_overlay_aligned",
        look_key=None,
        context_key="interpret/test_overlay_context",
        export=ExportSource(
            clips=lambda: clips,
            experiment_dir=lambda: experiment_dir,
            origin=lambda: ORIGIN,
            name=lambda: "Cluster #7",
            empty=lambda: "Nothing here.",
            experiment=lambda: None,
        ),
    )
    montage.set_context_s(0.0)
    montage.set_clips(
        [live_montage.MontageClip(c.end_t_ns - c.start_t_ns, c.rank, "") for c in clips]
    )
    montage._clock.stop()
    return montage


def _close(montage):
    montage.close()
    montage._worker.stop()
    montage.deleteLater()


def test_the_overlay_waits_for_a_run_that_can_align(qapp, tmp_path):
    montage = _grid(qapp, tmp_path)
    action = montage.exports.button.actions_by_kind["overlay"]
    montage.exports.sync()
    assert not action.isEnabled(), "the page has not said yet whether it can align"
    montage.set_alignment("No heading definition fits this skeleton.")
    montage.exports.sync()
    assert not action.isEnabled()
    assert action.toolTip().endswith("no heading definition fits this skeleton.")
    montage.set_alignment(None)
    montage.exports.sync()
    assert action.isEnabled() and "turned to face up" in action.toolTip()
    _close(montage)


def test_export_overlay_writes_a_still_for_single_frames(qapp, tmp_path):
    import cv2

    montage = _grid(qapp, tmp_path, spans=[(0, 0), (1, 1), (3, 3)])
    montage.set_alignment(None)
    exports = montage.exports
    out = tmp_path / "overlay.png"
    exports.export_overlay(out)
    assert drain_until(
        qapp, lambda: "Wrote overlay.png" in exports.written.text(), 20_000
    )
    picture = cv2.imread(str(out))
    assert picture is not None and picture.shape[0] == cm.OVERLAY_PX
    _close(montage)


def test_export_overlay_writes_a_video_of_the_picks_in_the_codec_asked(qapp, tmp_path):
    import av

    montage = _grid(qapp, tmp_path)
    montage.set_alignment(None)
    exports = montage.exports
    exports.start_picking()
    montage._clicked(1, NO_MODIFIER)
    montage._clicked(0, NO_MODIFIER)
    clips, _labels = exports.overlay_clips(exports.job())
    assert [c.rank for c in clips] == [2, 1], "the picks, in pick order"
    out = tmp_path / "overlay.mp4"
    exports.export_overlay(out, codec="mp4v")
    assert drain_until(
        qapp, lambda: "Wrote overlay.mp4" in exports.written.text(), 30_000
    )
    with av.open(str(out)) as container:
        frames = sum(1 for _ in container.decode(container.streams.video[0]))
    single = cm.ClusterMontage(exports.experiment_dir(), clips)
    try:
        single.prepare()
        assert frames == single.n_steps
    finally:
        single.close()
    assert not exports.is_picking(), "an export of the picks ends pick mode"
    _close(montage)


def test_an_overlay_of_every_clip_draws_the_grids_first_twelve(qapp, tmp_path):
    montage = _grid(qapp, tmp_path, spans=[(0, 1)] * 2 + [(1, 3)] * 14)
    clips, labels = montage.exports.overlay_clips(montage.exports.job())
    assert len(clips) == len(labels) == cm.OVERLAY_CLIPS == 12
    assert [c.rank for c in clips] == list(range(1, 13))
    _close(montage)
