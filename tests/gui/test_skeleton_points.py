"""The skeleton points picker (§22): any skeleton's points as checkboxes — the clip window's
**Trail points** — and the button that holds it in a menu.

Built on skeletons of different body plans on purpose: the widget must take whatever names it
is handed and assume none of its own.
"""

import pytest

pytest.importorskip("PyQt6")

from detail.gui.skeleton_points import (  # noqa: E402
    POINTS_TIP,
    SkeletonPointsButton,
    SkeletonPointsPicker,
)

MOUSE = ("nose", "neck_base", "mid_back", "tail_base", "left_forepaw", "right_forepaw")
FISH = ("snout", "l_fin", "r_fin", "caudal")


@pytest.fixture
def picker(qapp):
    return SkeletonPointsPicker()


def _untick(picker, name):
    picker._boxes[name].setChecked(False)


def test_every_point_is_ticked_and_reads_as_unset(picker):
    assert picker.value() is None and "No keypoints" in picker._note.text()
    picker.set_skeleton(MOUSE)
    assert picker.names() == list(MOUSE) and picker.value() is None
    assert picker._note.text() == "Every point leaves a trail."


def test_unticking_names_the_kept_points(picker):
    picker.set_skeleton(MOUSE)
    seen = []
    picker.changed.connect(lambda: seen.append(1))
    _untick(picker, "left_forepaw")
    _untick(picker, "right_forepaw")
    assert picker.value() == ["nose", "neck_base", "mid_back", "tail_base"]
    assert picker._note.text() == "Only the ticked points leave a trail." and seen
    for name in picker.names():
        _untick(picker, name)
    assert picker.value() == [] and "at least one" in picker._note.text()


def test_a_value_set_before_the_skeleton_is_held_and_applied(picker):
    picker.set_value(["snout", "caudal"])
    assert picker.value() == ["snout", "caudal"]  # reported back as it was set
    picker.set_skeleton(FISH)
    assert picker.value() == ["snout", "caudal"]
    assert not picker._boxes["l_fin"].isChecked()


def test_a_choice_carries_to_the_next_skeleton_for_the_names_they_share(picker):
    picker.set_skeleton(MOUSE)
    _untick(picker, "tail_base")
    picker.set_skeleton(("nose", "tail_base", "tail_tip"))
    assert picker.value() == ["nose", "tail_tip"]  # tail_base stays off, the new one on
    picker.set_value(None)
    assert picker.value() is None


def test_the_button_reaches_through_to_its_picker(qapp):
    button = SkeletonPointsButton("Trail points")
    seen = []
    button.changed.connect(lambda: seen.append(1))
    button.set_skeleton(FISH)
    assert button.value() is None and button.picker.names() == list(FISH)
    button.picker._boxes["l_fin"].setChecked(False)
    assert button.value() == ["snout", "r_fin", "caudal"] and seen
    assert button.menu() is not None, "the picker drops down from the button"


def test_the_button_is_greyed_only_with_a_reason(qapp):
    """`a_disabled_button_says_why`: the one way to disable the button carries the reason."""
    button = SkeletonPointsButton()
    assert button.isEnabled() and button.toolTip() == POINTS_TIP
    button.set_reason("This run used no keypoints.")
    assert not button.isEnabled()
    assert button.toolTip() == "This run used no keypoints."
    button.set_reason(None)
    assert button.isEnabled() and button.toolTip() == POINTS_TIP
