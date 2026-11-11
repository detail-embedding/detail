"""The clip window's Advanced player (``gui/advanced_player.py``) is the plain player over its
hooks and nothing else.

**The class this protects.** `_ClipPlayer` is embedded by eleven surfaces, none of which wants a
trail picker, three pictures or strips. #806 grew it by 283 lines behind a ``panels=`` flag;
here it grows a handful of hooks with today's behaviour as the default, and the Advanced
player overrides them. The first test is the structural half: the override set is exactly the
hooks (plus ``set_clip``, which loads the strips), so a popup-only feature that lands on the
base class is a failure here before it is a regression anywhere else. Rule R11 in
``docs/how_we_work.md``: replace the overrides with ``super()`` and these fail while
``test_behavior_query.py`` stays green.
"""

import inspect

import pytest

pytest.importorskip("PyQt6")

from detail.api.clip_views import ClipViewSettings  # noqa: E402
from detail.gui.advanced_player import AdvancedClipPlayer  # noqa: E402
from detail.gui.behavior_query import ClipSpec, _ClipPlayer  # noqa: E402
from tests.clip_fixtures import WALKER, walking  # noqa: E402

#: The seams the plain player opens, each with its own behaviour as the default.
HOOKS = (
    "_picture_widget",
    "_add_view_controls",
    "_decode",
    "_show_payload",
    "_show_unavailable",
    "_clip_only",
    "_redraw",
)


class _ExperimentAt:
    def __init__(self, path):
        self.dir = path


def test_the_advanced_player_overrides_the_hooks_and_nothing_else():
    overridden = {
        name
        for name, value in vars(AdvancedClipPlayer).items()
        if inspect.isfunction(value) and name in vars(_ClipPlayer)
    }
    assert overridden == set(HOOKS) | {"set_clip"}
    for name in HOOKS:
        assert inspect.isfunction(getattr(_ClipPlayer, name)), name


def test_the_plain_player_has_none_of_the_advanced_state(qapp, tmp_path):
    plain = _ClipPlayer(_ExperimentAt(tmp_path), autoplay=False, context_s=0)
    for name in (
        "_points",
        "_trail",
        "_trail_button",
        "_mask_trail",
        "_align",
        "_panes",
    ):
        assert not hasattr(plain, name), name
    assert plain._clip_only() is False, "with Walk edges off the whole window plays"


def test_view_settings_mirror_the_ticks(qapp, tmp_path):
    player = AdvancedClipPlayer(_ExperimentAt(tmp_path), autoplay=False, context_s=0)
    assert player.view_settings() == ClipViewSettings()
    seen = []
    player.view_changed.connect(lambda: seen.append(1))
    player._points.setChecked(True)
    player._trail.setChecked(True)
    player._mask_trail.setChecked(True)
    player._align.setChecked(True)
    player._masks.setChecked(False)
    got = player.view_settings()
    assert got == ClipViewSettings(
        masks_on_frame=False,
        points_on_frame=True,
        trail=True,
        trail_points=None,
        mask_trail=True,
        pose_aligned=True,
    )
    assert len(seen) == 5, "every tick announces itself to an enlarged pane"
    player._trail_button.picker.set_value(["nose"])  # before any skeleton: held
    assert player.view_settings().trail_points == ("nose",)


def test_loop_clip_plays_the_clip_alone(qapp, tmp_path):
    player = AdvancedClipPlayer(_ExperimentAt(tmp_path), autoplay=False, context_s=0)
    assert not player._clip_only()
    player._loop_clip.setChecked(True)
    assert player._clip_only() and player.scrubber()._loop


def test_set_clip_shows_the_panes_and_reads_the_strips_off_thread(qapp, tmp_path):
    from tests.gui.conftest import drain_until

    recording = walking(tmp_path)
    player = AdvancedClipPlayer(
        _ExperimentAt(recording.dir.parent.parent), autoplay=False, context_s=0
    )
    player.resize(900, 500)
    player.show()
    changed = []
    player.clip_changed.connect(lambda: changed.append(1))
    player.set_clip(ClipSpec(recording.id, 0, "emb_01", 0, int(4e9 / 30), "a walk"))
    assert changed == [1]
    assert drain_until(qapp, lambda: player.panes().masks.pixmap_size() is not None)
    assert drain_until(qapp, lambda: player.strips().curves() is not None)
    assert player.strips().names()[:2] == ["speed", "area"]
    assert player._trail_button.picker.names() == list(WALKER.names)
    assert player._points.isEnabled() and player._trail_button.isEnabled()
    player.close()
