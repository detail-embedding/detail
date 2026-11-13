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
"""

import pathlib
import tempfile

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

import detail.gui.clip_grid as clip_grid  # noqa: E402
from detail.gui.behavior_query import ClipSpec  # noqa: E402
from detail.gui.cluster_strip import ClusterClipStrip  # noqa: E402
from detail.gui.quick_preview import QuickPreview  # noqa: E402


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


def test_a_query_cell_marks_on_one_click_and_opens_on_two(qapp):
    """A single click marks and opens nothing; a second one opens, exactly once.

    Marking *also* happens on the way to a double-click, and that is Qt rather than a bug: the
    first half of a double-click is a click, and it lands on the clip you were pointing at
    anyway. An earlier version of this docstring claimed the opposite and the assertion below
    could not have caught it either way — which is why the property asserted is the one that can
    actually go wrong: a single click must not open a window, and a double must not open two.
    """
    cell = clip_grid._Cell(96)
    clip = _clips(1)[0]
    marked, opened = [], []
    cell.set_clip(clip, marked.append, on_double_click=opened.append)

    _shown(cell, qapp)

    QTest.mouseClick(cell, Qt.MouseButton.LeftButton)
    assert marked == [clip], "a click has to mark — `Cut here` acts on what is marked"
    assert opened == [], "a single click must not open a window"

    QTest.mouseDClick(cell, Qt.MouseButton.LeftButton)
    assert opened == [clip], "a double-click opens the clip once, not twice"


def test_a_query_cell_with_no_clip_opens_nothing(qapp):
    """An empty slot at the end of a short row is not a clip anybody can open."""
    cell = clip_grid._Cell(96)
    opened = []
    cell.set_clip(None, lambda _c: None, on_double_click=opened.append)
    _shown(cell, qapp)

    QTest.mouseDClick(cell, Qt.MouseButton.LeftButton)

    assert opened == []


def test_both_strips_wire_the_double_click(qapp):
    """Guard the guard for the split this closes.

    The two strips are edited at different times by whoever is looking at one of them, which is
    how the capability landed on one page four times before. Asserting the wiring over both means
    a third strip cannot quietly ship without it.
    """
    strip = _shown(ClusterClipStrip(_Experiment()), qapp)
    assert strip._cell.receivers(strip._cell.double_clicked) > 0

    grid = _shown(clip_grid.ClipGrid(_Experiment()), qapp)
    grid.show_clips(_clips(3))
    QApplication.processEvents()
    assert grid._cells[0]._on_double_click is not None


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


def test_a_query_cell_says_it_opens_only_where_a_strip_wired_it(qapp):
    """The same pair on the other strip, since the two are edited at different times."""
    cell = clip_grid._Cell(96)
    clip = _clips(1)[0]

    cell.set_clip(clip, lambda _c: None)
    assert cell.toolTip() == ""

    cell.set_clip(clip, lambda _c: None, on_double_click=lambda _c: None)
    assert "Double-click" in cell.toolTip()

    cell.set_clip(None, lambda _c: None, on_double_click=lambda _c: None)
    assert cell.toolTip() == "", "an empty slot is not a clip anybody can open"


def test_a_popup_with_a_sequence_steps_through_it(qapp, monkeypatch):
    """The labels ethogram opens a bout with every bout of its label one step away."""
    import detail.gui.behavior_query as behavior_query
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

    monkeypatch.setattr(behavior_query, "_ClipPlayer", _Player)
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
    import detail.gui.behavior_query as behavior_query
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

    monkeypatch.setattr(behavior_query, "_ClipPlayer", _Player)
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


def _open_on_frame_0(qapp, recording, *, track_id=0):
    from tests.gui.conftest import drain_until

    from detail.gui.clip_popup import ClipPopup

    clip = ClipSpec(recording.id, track_id, "emb_01", 0, int(2e9 / 30), "a clip")
    popup = ClipPopup(_ExperimentAt(recording.dir.parent.parent), clip, context_s=0)
    popup.resize(1200, 600)
    popup.show()
    player = popup.player()
    player.pause()
    assert drain_until(qapp, lambda: player._panes.masks.pixmap_size() is not None)
    return popup, player


def test_the_popup_shows_footage_masks_and_keypoints_side_by_side(qapp, tmp_path):
    """The footage on the left, the masks pane over the keypoints pane on the right, each
    decoded from the one frame — and a Keypoints toggle beside Masks for the footage."""
    from tests.api.test_clip_views import _two_animals, _with_keypoints

    from detail.gui.clip_panes import ClipPanes
    from detail.gui.image_view import ImageView

    recording = _with_keypoints(_two_animals(tmp_path))
    popup, player = _open_on_frame_0(qapp, recording)
    panes = player._panes
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
    from tests.api.test_clip_views import _two_animals

    recording = _two_animals(tmp_path)
    popup, player = _open_on_frame_0(qapp, recording)
    assert not player._points.isEnabled()
    assert "used no keypoints" in player._points.toolTip()
    assert player._panes.pose.text() == "No keypoints for this run"
    assert "used no keypoints" in player._panes.pose.toolTip()
    popup.close()


def test_the_keypoints_toggle_redraws_the_frame_on_screen(qapp, tmp_path):
    """Ticked, the footage gains the subject's points without a move of the playhead."""
    from tests.api.test_clip_views import _two_animals, _with_keypoints
    from tests.gui.conftest import drain_until

    recording = _with_keypoints(_two_animals(tmp_path))
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
    from tests.api.test_clip_views import _two_animals, _with_heading, _with_keypoints
    from tests.gui.conftest import drain_until

    recording = _with_heading(_with_keypoints(_two_animals(tmp_path)))
    popup, player = _open_on_frame_0(qapp, recording)
    assert player._align.isEnabled()
    pose = player._panes.pose
    before = pose.pixmap().toImage()
    player._align.setChecked(True)
    assert drain_until(qapp, lambda: pose.pixmap().toImage() != before)
    popup.close()

    bare = _with_keypoints(_two_animals(tmp_path / "bare"))
    popup, player = _open_on_frame_0(qapp, bare)
    assert not player._align.isEnabled()
    assert "No heading definition" in player._align.toolTip()
    popup.close()


# --- the strips under the slider ------------------------------------------------------- #
def _with_strips(qapp, recording, **kwargs):
    from tests.gui.conftest import drain_until

    popup, player = _open_on_frame_0(qapp, recording, **kwargs)
    assert drain_until(qapp, lambda: player.strips().signals() is not None)
    return popup, player


def test_the_strips_read_the_clips_window_under_the_slider(qapp, tmp_path):
    """Speed, area and the nearest animal, a quality row and a cluster row, on the frames the
    slider spans, with the clip shaded and the reading under the pointer in the tooltip.
    """
    from tests.api.test_clip_views import _two_animals

    popup, player = _with_strips(qapp, _two_animals(tmp_path))
    strips = player.strips()
    names = [s.name for s in strips.signals().signals]
    assert names == ["speed", "mask area", "to nearest animal"]
    assert len(strips.signals().frames) == player._len
    left, span = player.scrubber().handle_travel()
    rows = strips._rows()
    _kind, top, height = rows[2]  # to nearest animal
    tip = strips.describe(left, top + height / 2)  # frame 0: both animals there
    assert tip.startswith("to nearest animal:") and "body sizes" in tip
    _kind, top, height = rows[3]  # quality
    assert "not tracked" in strips.describe(
        left + span * 0.5, top + 1
    )  # frame 1 of 0..2
    popup.close()


def test_loop_clip_plays_the_clip_and_nothing_either_side(qapp, tmp_path):
    from tests.api.test_clip_views import _two_animals

    popup, player = _with_strips(qapp, _two_animals(tmp_path))
    player._loop_clip.setChecked(True)
    lo, hi = player.clip_span()
    first = player.window()[0]
    assert player.scrubber().playback_range() == (lo - first, hi - first)
    assert player.scrubber()._loop
    popup.close()


def test_the_trail_controls_follow_the_clips_keypoints(qapp, tmp_path):
    """With keypoints, Trail points lists the skeleton (the keypoint track form's picker,
    every point ticked) and a choice there redraws the frame on screen; the Trail tick says
    what it draws."""
    from tests.api.test_clip_views import WALKER, _walking
    from tests.gui.conftest import drain_until

    popup, player = _open_on_frame_0(qapp, _walking(tmp_path))
    picker = player._trail_picker
    assert player._trail_points.isEnabled()
    assert picker.names() == list(WALKER.names) and player.trail_points() is None
    player._trail.setChecked(True)
    assert drain_until(qapp, lambda: "keypoint" in player._trail.toolTip())
    before = player._src.pixmap().toImage()
    player.scrubber().set_frame(2)
    assert drain_until(qapp, lambda: player._src.pixmap().toImage() != before)
    walked = player._src.pixmap().toImage()
    picker.set_value(["left_paw"])
    assert player.trail_points() == ["left_paw"]
    assert drain_until(qapp, lambda: player._src.pixmap().toImage() != walked)
    popup.close()


def test_without_keypoints_the_points_choice_is_greyed_with_the_reason(qapp, tmp_path):
    from tests.api.test_clip_views import _walking
    from tests.gui.conftest import drain_until

    popup, player = _open_on_frame_0(qapp, _walking(tmp_path, keypoints=False))
    assert not player._trail_points.isEnabled()
    assert "used no keypoints" in player._trail_points.toolTip()
    assert "box" in player._trail_points.toolTip()
    player._trail.setChecked(True)
    assert drain_until(qapp, lambda: "box" in player._trail.toolTip())
    popup.close()


def test_the_mask_trail_tick_redraws_the_masks_pane(qapp, tmp_path):
    from tests.api.test_clip_views import _walking
    from tests.gui.conftest import drain_until

    popup, player = _open_on_frame_0(qapp, _walking(tmp_path, keypoints=False))
    masks = player._panes.masks
    first = masks.pixmap().toImage()
    player.scrubber().set_frame(2)
    assert drain_until(qapp, lambda: masks.pixmap().toImage() != first)
    before = masks.pixmap().toImage()
    player._mask_trail.setChecked(True)
    assert drain_until(qapp, lambda: masks.pixmap().toImage() != before)
    popup.close()


class _Labelling:
    """What a clustering page hands the window, without the clustering."""

    def __init__(self, experiment):
        self._experiment = experiment

    def title(self):
        return "cluster #7"

    def current_label(self):
        return ""

    def label_choices(self):
        return ["rearing"]

    def edit_reason(self):
        return ""

    def apply_label(self, name):
        pass

    def check(self, key, verdict):
        pass

    def node(self):
        return 7

    def load_checks(self):
        from detail.storage import member_checks

        return member_checks.summarize({})

    def experiment(self):
        return self._experiment

    def fingerprint(self, clip, progress):
        from types import SimpleNamespace

        return SimpleNamespace(rows=[])

    def clip_measures(self, clip, names):
        return {}

    def sibling(self):
        return None

    def member_clips(self, n):
        return []


#: The clip window's picture is at least this share of the window's body (everything between
#: the title bar and the Close button) at a realistic size, 1500 x 900, with a 4:3 frame. The
#: new layout measures 76% there (73% on Windows' own fonts at the opening size, and ~70% with a
#: 16:9 frame, which runs out of width first); the side-column layout it replaced measures 64%
#: at the same size even with the new thin strips, and held the picture to 21% (a 90px picture
#: in a 428px body) at its own opening size.
_PICTURE_SHARE = 0.70
#: A signal strip's most height: it was 30px, and is now about half.
_STRIP_ROW_MAX = 18


def test_the_picture_takes_most_of_the_window(qapp, tmp_path):
    """The point of the window is seeing the behaviour: the footage and its panes get the
    width and most of the height; the strips are thin, and labelling is one row under them.
    """
    from tests.api.test_clip_views import _walking
    from tests.gui.conftest import drain_until

    from detail.gui.clip_popup import ClipPopup
    from detail.gui.sizing import lines

    recording = _walking(tmp_path)
    experiment = _ExperimentAt(recording.dir.parent.parent)
    clip = ClipSpec(recording.id, 0, "emb_01", 0, int(4e9 / 30), "a clip")
    popup = ClipPopup(experiment, clip, context_s=0, labelling=_Labelling(experiment))
    popup.show()
    # After the show: the window opens at its content's size (`Dialog.showEvent`), and this
    # asks how a large window shares its room, whatever the fixture's small frame opens at.
    popup.resize(1500, 900)
    player = popup.player()
    player.pause()
    strips = player.strips()
    assert drain_until(qapp, lambda: strips.signals() is not None)
    assert drain_until(qapp, lambda: player._src.pixmap_size() is not None)
    QApplication.processEvents()

    body = popup._scroll.viewport()
    picture = (
        player._src
    )  # placed at the frame's own aspect: its height is drawn height
    share = picture.height() / body.height()
    assert share >= _PICTURE_SHARE, f"the picture is {share:.0%} of the window's body"

    rows = strips._rows()
    signal_rows = [h for kind, _top, h in rows if kind.startswith("signal:")]
    # A strip was 30px; now a line of fine print, about half that.
    assert signal_rows and max(signal_rows) <= _STRIP_ROW_MAX, "a strip is a thin row"
    assert strips.height() <= (len(signal_rows) + 2) * (_STRIP_ROW_MAX + 2)

    labelling = popup._labelling_row
    top = labelling.mapTo(popup, labelling.rect().topLeft()).y()
    strips_bottom = strips.mapTo(popup, strips.rect().bottomLeft()).y()
    assert top > strips_bottom, "the labelling sits under the strips"
    assert labelling.height() <= lines(2.5), "one row"
    assert labelling.width() >= 0.9 * body.width(), "a row across, not a side column"
    assert popup.fingerprint_panel() is None, "the cluster sections open on request"
    popup.close()


def test_cluster_details_open_under_the_labelling_row_and_are_remembered(
    qapp, tmp_path
):
    from tests.api.test_clip_views import _two_animals

    from detail.gui import clip_popup

    recording = _two_animals(tmp_path)
    experiment = _ExperimentAt(recording.dir.parent.parent)
    clip = ClipSpec(recording.id, 0, "emb_01", 0, int(2e9 / 30), "a clip")
    try:
        popup = clip_popup.ClipPopup(
            experiment, clip, context_s=0, labelling=_Labelling(experiment)
        )
        popup.show()
        popup._details_button.setChecked(True)
        assert popup.fingerprint_panel() is not None and popup.details_shown()
        assert clip_popup.remembered(clip_popup.DETAILS_KEY, False)
        again = clip_popup.ClipPopup(
            experiment, clip, context_s=0, labelling=_Labelling(experiment)
        )
        assert again.fingerprint_panel() is not None, "opens the way it was left"
        again.close()
        popup._details_button.setChecked(False)
        assert not popup.details_shown()
    finally:
        clip_popup.remember(clip_popup.DETAILS_KEY, False)
    popup.close()


def test_an_opener_s_lane_fills_the_cluster_row(qapp, tmp_path):
    from tests.api.test_clip_views import _two_animals
    from tests.gui.conftest import drain_until

    from detail.gui.clip_popup import ClipPopup

    asked = []

    def lane(clip, start_ns, end_ns):
        asked.append((clip.recording_id, start_ns, end_ns))
        return [
            {
                "start_ns": start_ns,
                "end_ns": end_ns,
                "kind": "this",
                "node": 7,
                "name": "",
            }
        ]

    recording = _two_animals(tmp_path)
    clip = ClipSpec(recording.id, 0, "emb_01", 0, int(2e9 / 30), "a clip")
    popup = ClipPopup(
        _ExperimentAt(recording.dir.parent.parent), clip, context_s=0, lane=lane
    )
    popup.show()
    strips = popup.player().strips()
    assert drain_until(qapp, strips.has_lane)
    assert asked[0][0] == recording.id
    assert "length" in popup._about.text()
    popup.close()


# --- Simple / Advanced, and blind mode ----------------------------------------------- #
def test_simple_view_is_the_plain_player_and_the_choice_is_remembered(qapp, tmp_path):
    from tests.api.test_clip_views import _two_animals

    from detail.gui import clip_popup

    recording = _two_animals(tmp_path)
    popup, player = _open_on_frame_0(qapp, recording)
    assert popup.is_advanced() and player._panes is not None
    try:
        popup._advanced.setChecked(False)
        simple = popup.player()
        assert simple is not player
        assert simple._panes is None and simple.strips() is None
        assert clip_popup.remembered(clip_popup.ADVANCED_KEY, True) is False
        clip = ClipSpec(recording.id, 0, "emb_01", 0, int(2e9 / 30), "a clip")
        again = clip_popup.ClipPopup(
            _ExperimentAt(recording.dir.parent.parent), clip, context_s=0
        )
        assert (
            not again.is_advanced()
        ), "the next window opens the way this one was left"
        again.close()
    finally:
        clip_popup.remember(clip_popup.ADVANCED_KEY, True)
    popup.close()


def test_blind_mode_names_recordings_by_letter(qapp, tmp_path):
    from tests.api.test_clip_views import _two_animals

    from detail.gui import clip_popup

    recording = _two_animals(tmp_path)
    popup, _player = _open_on_frame_0(qapp, recording)
    try:
        popup._blind.setChecked(True)
        assert popup.windowTitle() == "Clip 1 · recording A"
        assert recording.id not in popup._about.text()
    finally:
        clip_popup.remember(clip_popup.BLIND_KEY, False)
    popup.close()


def test_aliases_are_lettered_in_the_order_recordings_first_appear():
    from detail.gui.clip_popup import _aliases

    clips = [ClipSpec(r, 0, "e", 0, 1, "") for r in ("r9", "r2", "r9", "r5")]
    assert _aliases(clips) == {"r9": "A", "r2": "B", "r5": "C"}
    many = [ClipSpec(f"r{i}", 0, "e", 0, 1, "") for i in range(28)]
    assert _aliases(many)["r26"] == "AA" and _aliases(many)["r27"] == "AB"


# --- laptop sizes ----------------------------------------------------------------------- #
def test_the_window_is_no_taller_than_its_picture_and_controls(qapp, tmp_path):
    """The picture keeps its aspect, so a window opened at a fixed height is width-limited on a
    laptop and the player centred the picture in the spare height: at 1324x805, 73px empty over
    the picture and ~60px over Close. The height follows the width (`Dialog.content_height`).
    """
    from PyQt6.QtWidgets import QDialogButtonBox

    from tests.api.test_clip_views import _two_animals
    from tests.gui.conftest import drain_until

    from detail.gui.sizing import lines

    recording = _two_animals(tmp_path)
    popup, player = _open_on_frame_0(qapp, recording)
    try:
        assert drain_until(
            qapp,
            lambda: popup._follow_timer is None or not popup._follow_timer.isActive(),
        )
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


# --- non-modal, one per page, reused (11-9 meeting) ------------------------------------ #
def test_the_clip_window_is_non_modal_and_one_per_page_reused(qapp, tmp_path):
    from PyQt6 import sip

    from tests.api.test_clip_views import _two_animals, _with_keypoints
    from tests.gui.conftest import drain_until

    from detail.gui import clip_popup

    recording = _with_keypoints(_two_animals(tmp_path))
    experiment = _ExperimentAt(recording.dir.parent.parent)
    page = QWidget()
    first = ClipSpec(recording.id, 0, "emb_01", 0, int(2e9 / 30), "first")
    second = ClipSpec(recording.id, 1, "emb_01", 0, int(1e9 / 30), "second")
    window = clip_popup.open_clip_window(page, experiment, first, context_s=0)
    assert not window.isModal() and window.isVisible()
    assert window.parent() is page, "the page's child: it closes with the page"
    assert not window._position.isVisible(), "no sequence, no stepping"
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
    assert drain_until(qapp, lambda: sip.isdeleted(other)), "closes with its page"


class _NodeLabelling(_Labelling):
    def __init__(self, experiment, node):
        super().__init__(experiment)
        self._node = node

    def node(self):
        return self._node


def _no_lane(clip, start_ns, end_ns):
    return []


def test_a_reused_window_takes_the_next_clips_cluster_and_lane(qapp, tmp_path):
    """INTERPRET's next double-click may be on another cluster: the reused window labels and
    checks that one and draws its lane; the same cluster keeps its row (and its queue).
    """
    from tests.api.test_clip_views import _two_animals

    from detail.gui import clip_popup

    recording = _two_animals(tmp_path)
    experiment = _ExperimentAt(recording.dir.parent.parent)
    page = QWidget()
    clip = ClipSpec(recording.id, 0, "emb_01", 0, int(2e9 / 30), "a clip")
    window = clip_popup.open_clip_window(
        page,
        experiment,
        clip,
        context_s=0,
        advanced=True,
        labelling=_NodeLabelling(experiment, 7),
    )
    try:
        panel = window.label_panel()
        assert panel is not None
        clip_popup.open_clip_window(
            page, experiment, clip, context_s=0, labelling=_NodeLabelling(experiment, 7)
        )
        assert window.label_panel() is panel, "the same cluster keeps its row"
        clip_popup.open_clip_window(
            page,
            experiment,
            clip,
            context_s=0,
            labelling=_NodeLabelling(experiment, 9),
            lane=_no_lane,
        )
        assert window.label_panel() is not panel
        assert window._labelling.node() == 9
        assert window.player()._lane_source is _no_lane
        clip_popup.open_clip_window(page, experiment, clip, context_s=0)
        assert window.label_panel() is None, "an opener with no cluster has no row"
    finally:
        window.close()


class _Page(QWidget):
    """What `clip_labelling.PageLabelling` reads of INTERPRET, and its label queue."""

    def __init__(self, experiment):
        super().__init__()
        from types import SimpleNamespace

        self._experiment = experiment
        self._run = SimpleNamespace()
        self._terms = SimpleNamespace(unit="cluster")
        self._names = {"4": "groom", "7": ""}
        self._can_edit = True
        self.labelled = []

    def _edit_reason(self):
        return ""

    def _label(self, node, name, origin=None):
        self.labelled.append((node, name, origin))


def test_a_label_from_the_window_goes_through_the_pages_own_queue(qapp, tmp_path):
    """Labels written in the clip window are the page's: its ``_label`` — so its undo history
    and its Finish lock see them — told which cluster window asked."""
    from detail.gui.clip_labelling import PageLabelling

    page = _Page(_Experiment())
    origin = QWidget()
    labelling = PageLabelling(page, 7, origin=origin)
    assert labelling.title() == "cluster #7"
    assert labelling.label_choices() == ["groom"]
    labelling.apply_label("rear")
    assert page.labelled == [(7, "rear", origin)]
    assert labelling.edit_reason() == ""
    page._can_edit = False
    assert labelling.edit_reason() == "This run cannot be changed here."
    assert not hasattr(labelling, "open_profile"), "offered only where the page has one"


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

    counterexample = "from x import ClipPopup\nClipPopup(e, c, self).exec()"
    assert offenders(counterexample) == [
        "ClipPopup"
    ], "the scan finds a page building the window itself"
    bad = [
        path.relative_to(package).as_posix()
        for path in sorted(package.rglob("*.py"))
        if path.name != "clip_popup.py"
        and offenders(path.read_text(encoding="utf-8-sig"))
    ]
    assert not bad, f"{bad} build the clip window; call clip_popup.open_clip_window"


# --- Export ▾ and the orientation ------------------------------------------------------- #
def test_the_export_menu_offers_what_you_see_and_this_frame(qapp, tmp_path):
    from tests.api.test_clip_views import _two_animals, _with_heading, _with_keypoints

    recording = _with_heading(_with_keypoints(_two_animals(tmp_path)))
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
    from tests.api.test_clip_views import _two_animals, _with_heading, _with_keypoints
    from tests.gui.conftest import drain_until

    recording = _with_heading(_with_keypoints(_two_animals(tmp_path)))
    popup, _player = _open_on_frame_0(qapp, recording)
    target = tmp_path / "frame.png"
    popup.write_png(target)
    assert drain_until(qapp, lambda: target.exists() and popup._written.isVisible())
    assert "frame.png" in popup._written.text()
    popup.close()


def test_a_video_of_what_you_see_is_written_in_the_codec_chosen(
    qapp, tmp_path, monkeypatch
):
    """*Video of what you see…* asks for the codec in the save dialog (``ask_mp4_path``) and
    writes the window's picture in it, off the UI thread."""
    from tests.api.test_clip_views import _two_animals
    from tests.gui.conftest import drain_until

    from detail.api import clip_pane_export
    from detail.gui import animation_export

    recording = _two_animals(tmp_path)
    popup, _player = _open_on_frame_0(qapp, recording)
    target = tmp_path / "view.mp4"
    asked = []
    monkeypatch.setattr(
        animation_export,
        "ask_mp4_path",
        lambda parent, title, suggested: (target, "mp4v"),
    )
    real = clip_pane_export.export_clip_view

    def spy(*args, **kwargs):
        asked.append(kwargs["codec"])
        return real(*args, **kwargs)

    monkeypatch.setattr(clip_pane_export, "export_clip_view", spy)
    popup.export_video()
    assert drain_until(qapp, lambda: popup._written.isVisible())
    assert asked == ["mp4v"] and target.exists()
    popup.close()


def test_a_gif_of_what_you_see_is_written_at_the_size_and_rate_asked(
    qapp, tmp_path, monkeypatch
):
    """*GIF of what you see…* asks for a width and a rate, then a path, and writes the window's
    picture as a GIF off the UI thread. Its import of the GIF suffix pointed at a module that
    does not define it, so the item raised before any dialog showed."""
    from PIL import Image

    from tests.api.test_clip_views import _two_animals
    from tests.gui.conftest import drain_until

    from detail.gui import animation_export

    recording = _two_animals(tmp_path)
    popup, _player = _open_on_frame_0(qapp, recording)
    target = tmp_path / "view.gif"
    monkeypatch.setattr(
        animation_export, "ask_gif_settings", lambda parent, note: (96, 5.0)
    )
    monkeypatch.setattr(
        animation_export, "ask_save_path", lambda parent, title, suggested, kind: target
    )
    popup.export_gif()
    assert drain_until(qapp, lambda: popup._written.isVisible())
    with Image.open(target) as gif:
        assert gif.size[0] <= 96
    popup.close()


def test_the_orientation_switch_is_greyed_with_the_reason_and_turns_the_picture(
    qapp, tmp_path
):
    from tests.api.test_clip_views import _two_animals, _with_heading, _with_keypoints
    from tests.gui.conftest import drain_until

    from detail.gui import clip_popup

    recording = _with_keypoints(_two_animals(tmp_path))
    popup, player = _open_on_frame_0(qapp, recording)
    assert drain_until(
        qapp, lambda: popup.alignment_note() != clip_popup.ALIGN_CHECKING
    )
    assert "No heading definition" in popup._orientation.toolTip()
    assert not popup._orientation.isEnabled()
    popup.close()
    _with_heading(recording)
    popup, player = _open_on_frame_0(qapp, recording)
    assert drain_until(qapp, lambda: popup.alignment_note() is None)
    try:
        before = player._src.pixmap().toImage()
        size = player._src.pixmap_size()
        popup._orientation.activated.emit(1)  # "Aligned" from the menu
        assert popup.is_aligned() and player.view_settings().aligned
        assert clip_popup.remembered(clip_popup.ALIGNED_KEY, False) is True
        assert drain_until(qapp, lambda: player._src.pixmap().toImage() != before)
        assert player._src.pixmap_size() == size, "turned at the frame's own size"
    finally:
        clip_popup.remember(clip_popup.ALIGNED_KEY, False)
        popup.close()
