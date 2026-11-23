"""Double-clicking a thumbnail opens the clip properly, on both strips.

**The class this closes.** A strip's preview is three compromises made for *comparing* members
side by side: a square crop of the subject, a strided subset of its frames, and a 132px box. All
three are right for a row of them and wrong the moment somebody stops comparing and starts
looking — and until now there was no way out of them. The clustering strip played its members in
place and the query strip drew stills, which is the recorded cluster/query split
(`test_workspace_parity.py`) one capability further on.

**What each single click means is deliberately different, and that is the part to protect.** The
clustering strip has no player beside it, so a click plays the member where it sits. The query
strip's click already *marks* a clip — it is what `Cut here` acts on — and loads it into the full
player next to the row. Making that one play in place would take the gesture the cutoff flow
depends on. So the shared gesture is the **double**-click, and these assert that both strips have
it and that the single click still means what it meant.

The second half is the window itself (``gui/clip_popup.py``, ``gui/advanced_player.py``): the
three pictures, the view ticks and what greys them, the strips under the slider, Simple /
Advanced and Blind mode.
"""

import pathlib
import tempfile

from detail.gui.behavior_query import ClipSpec
from detail.gui.cluster_strip import ClusterClipStrip
from detail.gui.qt import (
    QApplication,
    QDialogButtonBox,
    QTest,
    QVBoxLayout,
    QWidget,
    Qt,
    receivers,
)
from detail.gui.quick_preview import QuickPreview
from detail.gui.sizing import lines
from tests.clip_fixtures import (
    WALKER,
    two_animals,
    walking,
    with_heading,
    with_keypoints,
)


class _Experiment:
    def __init__(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())


def _clips(n=6):
    return [
        ClipSpec("rec", 0, "emb_01", i * 1_000_000, i * 1_000_000 + 500_000, f"c{i}")
        for i in range(n)
    ]


def _shown(widget, qapp):
    host = QWidget()
    QVBoxLayout(host).addWidget(widget)
    host.resize(900, 300)
    host.show()
    QApplication.processEvents()
    widget._host = host  # keep it alive
    return widget


def _frame():
    import numpy as np

    return np.full((60, 80, 3), 90, dtype=np.uint8)


def test_the_quick_preview_asks_rather_than_opens(qapp):
    """It holds frames, not a clip — so which clip to open is the strip's knowledge, not its own.

    Asserted because the tempting shortcut is to give the preview the clip and let it open its
    own window, which would make a widget that is deliberately about *comparing* depend on the
    whole player stack.
    """
    preview = _shown(QuickPreview(96), qapp)
    seen = []
    preview.double_clicked.connect(lambda: seen.append(True))

    QTest.mouseDClick(preview, Qt.MouseButton.LeftButton)

    assert seen == [True]


def test_the_clustering_strip_opens_the_member_being_played(qapp, monkeypatch):
    """It plays in place on a single click, and a double-click opens *that* member."""
    opened = []
    monkeypatch.setattr(
        "detail.gui.clip_popup.open_clip_window",
        lambda opener, experiment, clip, **_kw: opened.append(clip),
    )
    strip = _shown(ClusterClipStrip(_Experiment()), qapp)
    clips = _clips()
    strip.show_clips(clips, tint=(200, 80, 40))
    QApplication.processEvents()

    strip._playing = 3  # the member the strip is playing in its cell
    strip._open_playing_clip()

    assert opened == [clips[3]], "opened a different member from the one playing"


def test_the_clustering_strip_opens_nothing_when_nothing_is_playing(qapp, monkeypatch):
    """The preview is hidden until a member is clicked, so a stray double-click has no subject."""
    opened = []
    monkeypatch.setattr(
        "detail.gui.clip_popup.open_clip_window",
        lambda opener, experiment, clip, **_kw: opened.append(clip),
    )
    strip = _shown(ClusterClipStrip(_Experiment()), qapp)
    strip.show_clips(_clips(), tint=None)

    strip._playing = None
    strip._open_playing_clip()

    assert opened == []


def test_the_cluster_strip_wires_the_double_click(qapp):
    """The double-click that opens a clip in its window is wired, not merely drawn.

    It was asserted over two strips while the query CALIBRATE had its own; that strip went with
    the move to the shared montage (2026-11), whose double-click is ``clip_activated``
    (``tests/gui/test_live_montage.py``).
    """
    strip = _shown(ClusterClipStrip(_Experiment()), qapp)
    assert receivers(strip._cell, strip._cell.double_clicked) > 0


def test_the_preview_says_it_opens_only_where_a_host_wired_it(qapp):
    """A gesture nobody can see is a capability with no way in.

    The cue is **computed from the connection**, not set beside it — the recorded rule for
    exactly this, because a claim and its wiring otherwise live in two files and one of them
    goes wrong. Asserted in both directions, since a tooltip that is always on is the same
    failure wearing the opposite sign: it promises an opener on a strip that has none.
    """
    unwired = _shown(QuickPreview(96), qapp)
    unwired.set_frames([_frame()], 40)
    assert unwired.toolTip() == "", "promised an opener nothing is connected to"

    wired = _shown(QuickPreview(96), qapp)
    wired.double_clicked.connect(lambda: None)
    wired.set_frames([_frame()], 40)
    assert "Double-click" in wired.toolTip()


def test_a_popup_with_a_sequence_steps_through_it(qapp, monkeypatch):
    """The labels ethogram opens a bout with every bout of its label one step away."""
    import detail.gui.advanced_player as advanced_player
    from detail.gui.clip_popup import ClipPopup

    shown = []

    class _Player(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def set_clip(self, clip):
            shown.append(clip.label)

        def focus_scrubber(self):
            pass

        def export_button(self):
            return QWidget()

        def set_aligned(self, _on):
            pass

    monkeypatch.setattr(advanced_player, "PlainClipPlayer", _Player)
    clips = _clips(3)
    popup = ClipPopup(
        _Experiment(), clips[1], sequence=clips, index=1, context_s=0, advanced=False
    )
    assert shown == ["c1"]
    popup.step(1)
    popup.step(1)  # clamped at the end
    popup.step(-2)
    assert shown == ["c1", "c2", "c2", "c0"] and popup.index == 0
    assert not popup._previous.isEnabled() and popup._next.isEnabled()
    assert popup._position.text() == "1 of 3"


def test_the_position_carries_the_openers_order_note(qapp, monkeypatch):
    import detail.gui.advanced_player as advanced_player
    from detail.gui.clip_popup import ClipPopup

    class _Player(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def set_clip(self, clip):
            pass

        def focus_scrubber(self):
            pass

        def export_button(self):
            return QWidget()

        def set_aligned(self, _on):
            pass

    monkeypatch.setattr(advanced_player, "PlainClipPlayer", _Player)
    second = 1_000_000_000
    clips = [
        ClipSpec("rec", 0, "emb_01", i * second, i * second + second // 2, f"c{i}")
        for i in range(3)
    ]
    popup = ClipPopup(
        _Experiment(),
        clips[0],
        sequence=clips,
        context_s=0,
        order_note="most typical first",
        advanced=False,
    )
    assert popup._position.text() == "1 of 3 (most typical first)"
    assert "length 0.5 s" in popup._about.text()


# --- the popup's three pictures ------------------------------------------------------- #
class _ExperimentAt:
    def __init__(self, path):
        self.dir = path


def _open_on_frame_0(qapp, recording, *, track_id=0, frames=2):
    from tests.gui.conftest import drain_until

    from detail.gui.clip_popup import ClipPopup

    clip = ClipSpec(
        recording.id, track_id, "emb_01", 0, int(frames * 1e9 / 30), "a clip"
    )
    popup = ClipPopup(
        _ExperimentAt(recording.dir.parent.parent), clip, context_s=0, advanced=True
    )
    popup.resize(1200, 600)
    popup.show()
    player = popup.player()
    player.pause()
    assert drain_until(qapp, lambda: player.panes().masks.pixmap_size() is not None)
    return popup, player


def test_the_popup_shows_footage_masks_and_keypoints_side_by_side(qapp, tmp_path):
    """The footage on the left, the masks pane over the keypoints pane on the right, each
    decoded from the one frame — and a Keypoints toggle beside Masks for the footage."""
    from detail.gui.clip_panes import ClipPanes
    from detail.gui.image_view import ImageView

    recording = with_keypoints(two_animals(tmp_path))
    popup, player = _open_on_frame_0(qapp, recording)
    panes = player.panes()
    assert isinstance(panes, ClipPanes)
    assert len(player.findChildren(ImageView)) == 3
    assert panes.masks.pixmap_size() == panes.pose.pixmap_size()
    frame = player._src.geometry()
    for side in (panes.masks, panes.pose):
        assert side.mapTo(panes, side.rect().topLeft()).x() >= frame.right()
    assert player._points.isEnabled()
    shot = tmp_path / "clip_popup.png"
    popup.grab().save(str(shot))
    assert shot.stat().st_size > 0
    popup.close()


def test_an_embedding_with_no_keypoints_greys_the_toggle_and_says_why(qapp, tmp_path):
    recording = two_animals(tmp_path)
    popup, player = _open_on_frame_0(qapp, recording)
    assert not player._points.isEnabled()
    assert "used no keypoints" in player._points.toolTip()
    assert player.panes().pose.text() == "No keypoints for this run"
    assert "used no keypoints" in player.panes().pose.toolTip()
    popup.close()


def test_the_keypoints_toggle_redraws_the_frame_on_screen(qapp, tmp_path):
    """Ticked, the footage gains the subject's points without a move of the playhead."""
    from tests.gui.conftest import drain_until

    recording = with_keypoints(two_animals(tmp_path))
    popup, player = _open_on_frame_0(qapp, recording)
    before = player._src.pixmap().toImage()
    player._points.setChecked(True)
    assert drain_until(qapp, lambda: player._src.pixmap().toImage() != before)
    popup.close()


def test_align_pose_turns_the_keypoints_pane_and_greys_without_a_heading(
    qapp, tmp_path
):
    """Ticked, the Keypoints pane is redrawn body-aligned on the frame already on screen; a
    run with no heading definition greys the tick with the reason."""
    from tests.gui.conftest import drain_until

    recording = with_heading(with_keypoints(two_animals(tmp_path)))
    popup, player = _open_on_frame_0(qapp, recording)
    assert player._align.isEnabled()
    pose = player.panes().pose
    before = pose.pixmap().toImage()
    player._align.setChecked(True)
    assert drain_until(qapp, lambda: pose.pixmap().toImage() != before)
    popup.close()

    bare = with_keypoints(two_animals(tmp_path / "bare"))
    popup, player = _open_on_frame_0(qapp, bare)
    assert not player._align.isEnabled()
    assert "No heading definition" in player._align.toolTip()
    popup.close()


# --- the strips under the slider ------------------------------------------------------- #
def _with_strips(qapp, recording, **kwargs):
    from tests.gui.conftest import drain_until

    popup, player = _open_on_frame_0(qapp, recording, **kwargs)
    assert drain_until(qapp, lambda: player.strips().curves() is not None)
    return popup, player


def test_the_strips_read_the_clips_window_under_the_slider(qapp, tmp_path):
    """Speed and area off the kinematics sidecar, on the frames the slider spans, with the
    reading under the pointer in the tooltip — and, with a heading, the body's shape too.
    """
    popup, player = _with_strips(qapp, walking(tmp_path, heading=True), frames=4)
    strips = player.strips()
    names = strips.names()
    assert names[:2] == ["speed", "area"] and "turning" in names
    assert strips.curves().window == player.window()
    left, span = player.scrubber().handle_travel()
    _kind, top, height = strips._rows()[0]
    tip = strips.describe(left + span * 0.5, top + height / 2)
    assert tip.startswith("speed:") and "px/s" in tip
    popup.close()


def test_without_a_sidecar_the_strips_say_how_to_get_one(qapp, tmp_path):
    popup, player = _with_strips(qapp, two_animals(tmp_path))
    strips = player.strips()
    assert strips.names() == []
    assert "backfill-kinematics" in strips.describe(10, 3)
    popup.close()


def test_loop_clip_plays_the_clip_and_nothing_either_side(qapp, tmp_path):
    popup, player = _with_strips(qapp, two_animals(tmp_path))
    player._loop_clip.setChecked(True)
    lo, hi = player.clip_span()
    first = player.window()[0]
    assert player.scrubber().playback_range() == (lo - first, hi - first)
    assert player.scrubber()._loop
    popup.close()


def test_the_trail_controls_follow_the_clips_keypoints(qapp, tmp_path):
    """With keypoints, Trail points lists the skeleton (every point ticked) and a choice there
    redraws the frame on screen; the Trail tick says what it draws."""
    from tests.gui.conftest import drain_until

    popup, player = _open_on_frame_0(qapp, walking(tmp_path), frames=4)
    picker = player._trail_button.picker
    assert player._trail_button.isEnabled()
    assert picker.names() == list(WALKER.names) and player.trail_points() is None
    player._trail.setChecked(True)
    assert drain_until(qapp, lambda: "keypoint" in player._trail.toolTip())
    before = player._src.pixmap().toImage()
    player.scrubber().set_frame(2)
    assert drain_until(qapp, lambda: player._src.pixmap().toImage() != before)
    walked = player._src.pixmap().toImage()
    picker.set_value(["left_paw"])
    assert player.trail_points() == ["left_paw"]
    assert player.view_settings().trail_points == ("left_paw",)
    assert drain_until(qapp, lambda: player._src.pixmap().toImage() != walked)
    popup.close()


def test_without_keypoints_the_points_choice_is_greyed_with_the_reason(qapp, tmp_path):
    from tests.gui.conftest import drain_until

    popup, player = _open_on_frame_0(qapp, walking(tmp_path, keypoints=False), frames=4)
    assert not player._trail_button.isEnabled()
    assert "used no keypoints" in player._trail_button.toolTip()
    assert "centre" in player._trail_button.toolTip()
    player._trail.setChecked(True)
    assert drain_until(qapp, lambda: "centre of the animal" in player._trail.toolTip())
    popup.close()


def test_the_mask_trail_tick_redraws_the_masks_pane(qapp, tmp_path):
    from tests.gui.conftest import drain_until

    popup, player = _open_on_frame_0(qapp, walking(tmp_path, keypoints=False), frames=4)
    masks = player.panes().masks
    first = masks.pixmap().toImage()
    player.scrubber().set_frame(2)
    assert drain_until(qapp, lambda: masks.pixmap().toImage() != first)
    before = masks.pixmap().toImage()
    player._mask_trail.setChecked(True)
    assert drain_until(qapp, lambda: masks.pixmap().toImage() != before)
    popup.close()


#: The clip window's picture is at least this share of the window's body (everything between
#: the title bar and the Close button) at a realistic size, 1500 x 900, with a 4:3 frame. The
#: layout measures ~76% there; the side-column layout it replaced measured 64% at the same size
#: and held the picture to 21% at its own opening size.
_PICTURE_SHARE = 0.70
#: A strip's most height: it was 30px, and is now about half.
_STRIP_ROW_MAX = 20


def test_the_picture_takes_most_of_the_window(qapp, tmp_path):
    """The point of the window is seeing the behaviour: the footage and its panes get the
    width and most of the height; the strips are thin rows under the slider."""
    from tests.gui.conftest import drain_until

    from detail.gui.clip_popup import ClipPopup

    recording = walking(tmp_path)
    clip = ClipSpec(recording.id, 0, "emb_01", 0, int(4e9 / 30), "a clip")
    popup = ClipPopup(
        _ExperimentAt(recording.dir.parent.parent), clip, context_s=0, advanced=True
    )
    popup.resize(1500, 900)
    popup.show()
    player = popup.player()
    player.pause()
    strips = player.strips()
    assert drain_until(qapp, lambda: strips.curves() is not None)
    assert drain_until(qapp, lambda: player._src.pixmap_size() is not None)
    QApplication.processEvents()

    body = popup._scroll.viewport()
    share = player._src.height() / body.height()
    assert share >= _PICTURE_SHARE, f"the picture is {share:.0%} of the window's body"

    rows = strips._rows()
    curve_rows = [h for kind, _top, h in rows if kind.startswith("curve:")]
    assert curve_rows and max(curve_rows) <= _STRIP_ROW_MAX, "a strip is a thin row"
    assert strips.height() <= (len(curve_rows) + 2) * (_STRIP_ROW_MAX + 2)
    popup.close()


# --- Simple / Advanced, and blind mode ----------------------------------------------- #
def test_simple_view_is_the_plain_player_and_the_choice_is_remembered(qapp, tmp_path):
    from detail.gui import clip_popup
    from detail.gui.advanced_player import AdvancedClipPlayer

    recording = two_animals(tmp_path)
    popup, player = _open_on_frame_0(qapp, recording)
    assert popup.is_advanced() and isinstance(player, AdvancedClipPlayer)
    try:
        popup._advanced.setChecked(False)
        simple = popup.player()
        assert simple is not player
        assert not isinstance(simple, AdvancedClipPlayer)
        assert not hasattr(simple, "_panes") and not hasattr(simple, "_strips")
        assert clip_popup.remembered(clip_popup.ADVANCED_KEY, True) is False
        clip = ClipSpec(recording.id, 0, "emb_01", 0, int(2e9 / 30), "a clip")
        again = clip_popup.ClipPopup(
            _ExperimentAt(recording.dir.parent.parent), clip, context_s=0
        )
        assert not again.is_advanced(), "the next window opens the way this was left"
        again.close()
    finally:
        clip_popup.remember(clip_popup.ADVANCED_KEY, True)
    popup.close()


def test_blind_mode_names_recordings_by_letter(qapp, tmp_path):
    from detail.gui import clip_popup

    recording = two_animals(tmp_path)
    popup, _player = _open_on_frame_0(qapp, recording)
    try:
        popup._blind.setChecked(True)
        assert popup.windowTitle() == "Clip 1 · recording A"
        assert recording.id not in popup._about.text()
        assert clip_popup.remembered(clip_popup.BLIND_KEY, False) is True
        popup._blind.setChecked(False)
        assert popup.windowTitle() == "a clip"
    finally:
        clip_popup.remember(clip_popup.BLIND_KEY, False)
    popup.close()


def test_aliases_are_lettered_in_the_order_recordings_first_appear():
    from detail.gui.clip_popup import _aliases

    clips = [ClipSpec(r, 0, "e", 0, 1, "") for r in ("r9", "r2", "r9", "r5")]
    assert _aliases(clips) == {"r9": "A", "r2": "B", "r5": "C"}
    many = [ClipSpec(f"r{i}", 0, "e", 0, 1, "") for i in range(28)]
    assert _aliases(many)["r26"] == "AA" and _aliases(many)["r27"] == "AB"


# --- laptop sizes (the 1440x847 render audit) ---------------------------------------- #
def _flow_lines(player) -> dict:
    """``{top y: [widget, ...]}`` for the visible units of the scrubber's controls row."""
    flow = player._scrubber._controls
    out: dict = {}
    for i in range(flow.count()):
        widget = flow.itemAt(i).widget()
        if widget is not None and not widget.isHidden():
            out.setdefault(widget.geometry().top(), []).append(widget)
    return out


def test_the_frame_counter_never_wraps_onto_a_line_by_itself(qapp, tmp_path):
    """At 1324px (92% of a 1440 laptop screen) the Advanced row did not fit, and the one thing
    that wrapped was the frame counter, alone on a second line. The view ticks are a unit of
    their own now: when the row is short, they move down together and the counter ends their
    line."""
    recording = two_animals(tmp_path)
    popup, player = _open_on_frame_0(qapp, recording)
    try:
        flow = player._scrubber._controls
        one_line = flow.sizeHint().width()
        margins = popup.width() - player._scrubber.width()
        popup.resize(one_line + margins - 60, popup.height())
        qapp.processEvents()
        qapp.processEvents()
        rows = _flow_lines(player)
        assert len(rows) >= 2, "the row must wrap for the check to bite"
        readout = player._scrubber._readout
        line = rows[readout.geometry().top()]
        assert len(line) > 1, "the frame counter is alone on its line"
        assert player.view_ticks in line
    finally:
        popup.close()


def test_the_window_is_no_taller_than_its_picture_and_controls(qapp, tmp_path):
    """The picture keeps its aspect, so a window opened at a fixed height is width-limited on a
    laptop and the player centred the picture in the spare height: at 1324x805, 73px empty over
    the picture and ~60px over Close. The height now follows the width."""
    from tests.gui.conftest import drain_until

    recording = two_animals(tmp_path)
    popup, player = _open_on_frame_0(qapp, recording)
    try:
        # Settle the opening fit first, so nothing pending from it can do the work this test
        # is about: it passed only when an unrelated layout request arrived after the resize.
        assert drain_until(qapp, lambda: not popup._fit_timer.isActive())
        for _ in range(3):
            qapp.processEvents()
        width = 1300 if popup.width() != 1300 else 1260
        popup.resize(width, 1000)
        assert popup.height() == 1000, "the resize itself is taken as asked"
        assert drain_until(qapp, lambda: popup.height() < 1000), "a new width refits"
        assert popup.width() == width
        qapp.processEvents()
        picture = player._block.picture
        about = popup._about
        gap_over = (
            picture.mapTo(popup, picture.rect().topLeft()).y()
            - about.mapTo(popup, about.rect().bottomLeft()).y()
        )
        scrubber = player._scrubber
        close = popup.findChildren(QDialogButtonBox)[0]
        gap_under = (
            close.mapTo(popup, close.rect().topLeft()).y()
            - scrubber.mapTo(popup, scrubber.rect().bottomLeft()).y()
        )
        assert gap_over <= lines(1.5), gap_over
        assert gap_under <= lines(2), gap_under
    finally:
        popup.close()


def test_the_window_opens_nearly_screen_wide_and_capped_in_height():
    """A viewer on a laptop is short of width: it opens at 98% of the screen's free width (the
    Advanced controls fit one line there) and 92% of its height, then trims the height.
    """
    from detail.gui.qt import QRect

    from detail.gui.clip_popup import _TALL, _WIDE, opening_size

    class _Screen:
        def availableGeometry(self):
            return QRect(0, 0, 1440, 875)

    size = opening_size(_Screen())
    assert size.width() == int(1440 * 0.98) or size.width() == lines(_WIDE)
    assert size.height() == min(int(875 * 0.92), lines(_TALL))


# --- non-modal, one per page, reused (11-9 meeting) ------------------------------------ #
def test_the_clip_window_is_non_modal_and_one_per_page_reused(qapp, tmp_path):
    from detail.gui import clip_popup

    recording = with_keypoints(two_animals(tmp_path))
    experiment = _ExperimentAt(recording.dir.parent.parent)
    page = QWidget()
    first = ClipSpec(recording.id, 0, "emb_01", 0, int(2e9 / 30), "first")
    second = ClipSpec(recording.id, 1, "emb_01", 0, int(1e9 / 30), "second")
    window = clip_popup.open_clip_window(page, experiment, first, context_s=0)
    assert not window.isModal() and window.isVisible()
    assert window.parent() is page, "the page's child: it closes with the page"
    again = clip_popup.open_clip_window(
        page, experiment, second, context_s=0, sequence=[first, second], index=1
    )
    assert again is window, "the second double-click reuses the window"
    assert window.windowTitle() == "second"
    assert window._position.isVisible() and window._position.text() == "2 of 2"
    elsewhere = QWidget()
    other = clip_popup.open_clip_window(elsewhere, experiment, first, context_s=0)
    assert other is not window, "another page has its own"
    window.close()
    qapp.processEvents()
    fresh = clip_popup.open_clip_window(page, experiment, first, context_s=0)
    assert fresh is not window
    fresh.close()
    elsewhere.deleteLater()
    from tests.gui.conftest import drain_until
    from detail.gui.qt import is_deleted

    assert drain_until(qapp, lambda: is_deleted(other)), "closes with its page"


def test_every_opener_goes_through_the_one_non_modal_opener():
    """Guard: no module builds the clip window itself, or opens one modally — that is how four
    pages each grew their own `.exec()` and the window blocked every one of them. The scan is
    checked against a counterexample so it cannot pass by matching nothing."""
    import ast

    import detail.gui as gui

    package = pathlib.Path(gui.__file__).parent

    def offenders(source: str) -> list[str]:
        found = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else getattr(func, "id", "")
                )
                if name == "ClipPopup":
                    found.append(name)
        return found

    assert offenders("from x import ClipPopup\nClipPopup(e, c, self).exec()") == [
        "ClipPopup"
    ], "the scan finds a page building the window itself"
    bad = [
        path.relative_to(package).as_posix()
        for path in sorted(package.rglob("*.py"))
        if path.name != "clip_popup.py"
        and offenders(path.read_text(encoding="utf-8-sig"))
    ]
    assert not bad, f"{bad} build the clip window; call clip_popup.open_clip_window"


def test_the_export_menu_offers_what_you_see_and_this_frame(qapp, tmp_path):
    recording = with_heading(with_keypoints(two_animals(tmp_path)))
    popup, player = _open_on_frame_0(qapp, recording)
    texts = [a.text() for a in popup.export_menu().actions() if a.text()]
    assert texts == [
        "Video of what you see…",
        "GIF of what you see…",
        "This frame as PNG…",
    ]
    assert not player.export_button().isVisible(), "one export in the window, not two"
    settings, panes = popup.export_settings()
    assert panes and settings == player.view_settings()
    assert popup.export_frames() == player.window()
    popup.close()


def test_this_frame_as_png_writes_off_the_ui_thread(qapp, tmp_path):
    from tests.gui.conftest import drain_until

    recording = with_heading(with_keypoints(two_animals(tmp_path)))
    popup, _player = _open_on_frame_0(qapp, recording)
    target = tmp_path / "frame.png"
    popup.write_png(target)
    assert drain_until(qapp, lambda: target.exists() and popup._written.isVisible())
    assert "frame.png" in popup._written.text()
    popup.close()


def test_the_orientation_switch_is_greyed_with_the_reason_and_turns_the_picture(
    qapp, tmp_path
):
    from tests.gui.conftest import drain_until

    from detail.gui import clip_popup

    recording = with_keypoints(two_animals(tmp_path))
    popup, player = _open_on_frame_0(qapp, recording)
    assert drain_until(
        qapp, lambda: popup.alignment_note() != clip_popup.ALIGN_CHECKING
    )
    assert "No heading definition" in popup._orientation.toolTip()
    assert not popup._orientation.isEnabled()
    popup.close()
    with_heading(recording)
    popup, player = _open_on_frame_0(qapp, recording)
    assert drain_until(qapp, lambda: popup.alignment_note() is None)
    try:
        before = player._src.pixmap_size()
        popup._orientation.activated.emit(1)  # "Aligned" from the menu
        assert popup.is_aligned() and player.view_settings().aligned
        assert clip_popup.remembered(clip_popup.ALIGNED_KEY, False) is True
        assert player._src.pixmap_size() == before, "turned at the frame's own size"
    finally:
        clip_popup.remember(clip_popup.ALIGNED_KEY, False)
        popup.close()
