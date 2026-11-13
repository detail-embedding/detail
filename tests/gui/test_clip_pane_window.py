"""Clicking the clip window's Masks or Keypoints pane opens it large (``gui/clip_pane_window.py``).

Pinned:

- **The cue is derived from the wiring.** The pointing hand and "Click to enlarge" appear on the
  side panes exactly while something listens to ``enlarge_wanted`` — a ``ClipPanes`` nobody wired
  advertises nothing (the negative case), and the clip window's does.
- **The window is the pane at full size**, not the thumbnail scaled: its picture is the frame's
  own resolution.
- **It follows the clip window live** — a tick there changes what this one draws — and starts on
  the frame the clip window was on.
- **Its exports** write that pane, at the size and rate asked for.
- **One window per pane, and Close ends it**: a second click raises the open window, and
  Close pauses its playback, unhooks it from the player and deletes it. Before, every click
  built a window and Close only hid it, still following the player and decoding in a loop.
"""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402

from detail.gui.behavior_query import ClipSpec  # noqa: E402
from tests.api.test_clip_views import H, W, _walking, count_decodes  # noqa: E402
from tests.gui.conftest import drain_until  # noqa: E402


class _ExperimentAt:
    def __init__(self, path):
        self.dir = path


def _popup(qapp, tmp_path, *, keypoints=True):
    from detail.gui.clip_popup import ClipPopup

    recording = _walking(tmp_path, keypoints=keypoints)
    clip = ClipSpec(recording.id, 0, "emb_01", 0, 4 * int(1e9 / 30), "a walk")
    popup = ClipPopup(
        _ExperimentAt(recording.dir.parent.parent), clip, context_s=0, advanced=True
    )
    popup.resize(1200, 700)
    popup.show()
    player = popup.player()
    player.pause()
    assert drain_until(qapp, lambda: player._panes.masks.pixmap_size() is not None)
    return popup, player


def _window(qapp, player, pane):
    window = player.open_pane(pane)
    window.scrubber().pause()
    assert drain_until(qapp, lambda: window.view().pixmap_size() is not None)
    return window


def _gone(window) -> bool:
    """Closed and deleted (``WA_DeleteOnClose``), or at least hidden."""
    try:
        return not window.isVisible()
    except RuntimeError:  # the C++ side is gone: deleted on close, as it should be
        return True


def test_an_unwired_pane_offers_no_click(qapp):
    """The negative case: a ``ClipPanes`` with nothing listening shows no hand and no hint."""
    from detail.gui.clip_panes import ENLARGE_TIP, ClipPanes
    from detail.gui.image_view import ImageView

    panes = ClipPanes(ImageView(zoomable=True))
    assert not panes.enlargeable()
    assert panes.masks.rest_cursor() is None
    assert ENLARGE_TIP not in panes.masks.toolTip()
    panes.enlarge_wanted.connect(lambda _pane: None)
    assert panes.enlargeable()
    assert panes.masks.rest_cursor() == Qt.CursorShape.PointingHandCursor
    assert ENLARGE_TIP in panes.pose.toolTip()


def test_the_clip_windows_panes_say_click_to_enlarge(qapp, tmp_path):
    from detail.gui.clip_panes import ENLARGE_TIP

    popup, player = _popup(qapp, tmp_path)
    panes = player._panes
    for pane in (panes.masks, panes.pose):
        assert pane.rest_cursor() == Qt.CursorShape.PointingHandCursor
        assert ENLARGE_TIP in pane.toolTip()
    popup.close()


def test_a_click_on_a_pane_opens_it_full_size(qapp, tmp_path):
    from PyQt6.QtTest import QTest

    from detail.gui.clip_pane_window import ClipPaneWindow

    popup, player = _popup(qapp, tmp_path)
    QTest.mouseClick(player._panes.masks, Qt.MouseButton.LeftButton)
    assert drain_until(qapp, lambda: popup.findChildren(ClipPaneWindow))
    (window,) = popup.findChildren(ClipPaneWindow)
    assert window.isVisible() and not window.isModal()
    window.scrubber().pause()
    assert drain_until(qapp, lambda: window.view().pixmap_size() is not None)
    size = window.view().pixmap_size()
    assert (size.width(), size.height()) == (W, H), "the frame's own resolution"
    assert "Masks" in window.windowTitle()
    popup.close()


def test_the_enlarged_pane_follows_the_clip_windows_ticks(qapp, tmp_path):
    popup, player = _popup(qapp, tmp_path)
    player._scrubber.set_frame(4)
    assert drain_until(qapp, lambda: player.current_frame() == 4)
    window = _window(qapp, player, "masks")
    assert window.current_frame() == 4, "it opens on the clip window's frame"
    assert not window.settings().mask_trail
    before = window.view().pixmap().toImage()
    player._mask_trail.setChecked(True)
    assert window.settings().mask_trail, "a tick in the clip window reaches it"
    assert drain_until(qapp, lambda: window.view().pixmap().toImage() != before), (
        "and it redraws with it"
    )  # fmt: skip
    player._trail.setChecked(True)
    assert window.settings().trail
    popup.close()


def test_the_enlarged_pane_decodes_no_video(qapp, tmp_path, monkeypatch):
    """It shows the masks or the keypoints, never the footage, so drawing it opens no video.
    Counted on this thread only: the clip window behind it still decodes its own footage.
    """
    import threading

    popup, player = _popup(qapp, tmp_path)
    player._mask_trail.setChecked(True)
    player._trail.setChecked(True)
    for pane in ("masks", "keypoints"):
        window = _window(qapp, player, pane)
        calls = count_decodes(monkeypatch)
        for step in range(5):
            views, reason = window._compute(step)
            assert reason is None and views is not None
        here = threading.get_ident()
        assert [c for c in calls if c[1] == here] == [], pane
        monkeypatch.undo()
        window.close()
    popup.close()


def test_the_keypoints_pane_without_keypoints_says_so(qapp, tmp_path):
    popup, player = _popup(qapp, tmp_path, keypoints=False)
    window = player.open_pane("keypoints")
    window.scrubber().pause()
    assert drain_until(qapp, lambda: "No keypoints" in (window.view().text() or ""))
    popup.close()


def test_export_gif_writes_the_pane_at_the_size_and_rate_asked(
    qapp, tmp_path, monkeypatch
):
    from detail.gui import clip_pane_window

    popup, player = _popup(qapp, tmp_path)
    window = _window(qapp, player, "masks")
    out = tmp_path / "out" / "masks.gif"
    asked = []
    monkeypatch.setattr(
        clip_pane_window,
        "ask_gif_settings",
        lambda parent, note: asked.append(note) or (40, 10.0),
    )
    monkeypatch.setattr(clip_pane_window, "ask_save_path", lambda *a: out)
    window.export_gif()
    assert drain_until(qapp, lambda: "Wrote masks.gif" in window._status.text(), 20_000)
    assert "The clip itself" in asked[0], "the dialog says which frames it writes"
    with Image.open(out) as gif:
        assert gif.size[0] == 40 and gif.info["duration"] == 100
        assert gif.n_frames == 2  # 0 and 100 ms of a 133 ms clip
    popup.close()


@pytest.mark.parametrize("codec", ["h264", "mp4v"])
def test_export_video_writes_every_frame_at_full_size(
    qapp, tmp_path, monkeypatch, codec
):
    """The codec chosen as the save dialog's file type is the one in the file."""
    import av
    import cv2

    from detail.gui import clip_pane_window
    from detail.storage import video as video_store

    popup, player = _popup(qapp, tmp_path)
    window = _window(qapp, player, "keypoints")
    out = tmp_path / "pose.mp4"
    monkeypatch.setattr(clip_pane_window, "ask_mp4_path", lambda *a: (out, codec))
    window.export_video()
    assert drain_until(qapp, lambda: "Wrote pose.mp4" in window._status.text(), 20_000)
    cap = cv2.VideoCapture(str(out))
    try:
        ok, first = cap.read()
        count = 1 + sum(1 for _ in iter(lambda: cap.read()[0], False))
    finally:
        cap.release()
    assert ok and first.shape[:2] == (H, W) and count == 5
    assert (first > 230).all(axis=2).mean() > 0.5, "the pose on white"
    assert np.asarray(first).std() > 0
    with av.open(str(out)) as container:
        name = container.streams.video[0].codec_context.name
    written = video_store.written_codec(codec)
    assert name == {"h264": "h264", "mp4v": "mpeg4"}[written]
    popup.close()


def test_the_enlarged_pane_closes_with_the_player_it_follows(qapp, tmp_path):
    """Unticking Advanced rebuilds the clip window's player; a window following the old one
    would follow nothing, so it closes."""
    popup, player = _popup(qapp, tmp_path)
    window = _window(qapp, player, "masks")
    popup._advanced.setChecked(False)
    assert drain_until(qapp, lambda: _gone(window))
    popup._advanced.setChecked(True)
    popup.close()


def test_a_second_click_raises_the_open_window_rather_than_opening_another(
    qapp, tmp_path
):
    popup, player = _popup(qapp, tmp_path)
    first = _window(qapp, player, "masks")
    again = player.open_pane("masks")
    assert again is first and player.pane_windows == {"masks": first}
    other = player.open_pane("keypoints")
    assert other is not first and set(player.pane_windows) == {"masks", "keypoints"}
    first.close()
    assert drain_until(qapp, lambda: "masks" not in player.pane_windows)
    fresh = player.open_pane("masks")
    assert fresh is not first and player.pane_windows["masks"] is fresh
    popup.close()


def test_closing_the_window_stops_its_playback_and_unhooks_it(
    qapp, tmp_path, monkeypatch
):
    popup, player = _popup(qapp, tmp_path)
    before = player.receivers(player.view_changed)
    window = _window(qapp, player, "masks")
    assert player.receivers(player.view_changed) == before + 1
    paused = []
    monkeypatch.setattr(window.scrubber(), "pause", lambda: paused.append(1))
    window.close()
    assert paused, "Close pauses the loop that would otherwise decode forever"
    assert drain_until(qapp, lambda: _gone(window))
    assert player.receivers(player.view_changed) == before, "unhooked from the player"
    assert "masks" not in player.pane_windows
    player._mask_trail.setChecked(True)  # reaches no dead window
    popup.close()
