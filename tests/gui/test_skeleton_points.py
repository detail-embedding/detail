"""The skeleton points picker (§26): any skeleton's points as checkboxes, for ``track_points``.

Built on skeletons of different body plans on purpose: the widget must take whatever names and
pairs it is handed and assume none of its own.
"""

import pytest

from detail.gui.skeleton_points import SkeletonPointsPicker

MOUSE = (
    ("nose", "neck_base", "mid_back", "tail_base", "left_forepaw", "right_forepaw"),
    (("left_forepaw", "right_forepaw"),),
)
FISH = (("snout", "l_fin", "r_fin", "caudal"), (("l_fin", "r_fin"),))


@pytest.fixture
def picker(qapp):
    return SkeletonPointsPicker()


def _untick(picker, name):
    picker._boxes[name].setChecked(False)


def test_every_point_is_ticked_and_reads_as_unset(picker):
    assert picker.value() is None and "detect run" in picker._note.text()
    picker.set_skeleton(*MOUSE)
    assert picker.names() == list(MOUSE[0]) and picker.value() is None
    assert picker._note.text() == "Every point is tracked."


def test_unticking_names_the_kept_points_and_a_split_pair_is_called_out(picker):
    picker.set_skeleton(*MOUSE)
    seen = []
    picker.changed.connect(lambda: seen.append(1))
    _untick(picker, "left_forepaw")
    _untick(picker, "right_forepaw")
    assert picker.value() == ["nose", "neck_base", "mid_back", "tail_base"]
    assert "detect run's placement" in picker._note.text() and seen
    picker._boxes["right_forepaw"].setChecked(True)
    assert "left_forepaw / right_forepaw" in picker._note.text()
    for name in picker.names():
        _untick(picker, name)
    assert picker.value() == [] and "at least one" in picker._note.text()


def test_a_value_set_before_the_skeleton_is_held_and_applied(picker):
    picker.set_value(["snout", "caudal"])
    assert picker.value() == ["snout", "caudal"]  # reported back as it was set
    picker.set_skeleton(*FISH)
    assert picker.value() == ["snout", "caudal"]
    assert not picker._boxes["l_fin"].isChecked()


def test_a_choice_carries_to_the_next_skeleton_for_the_names_they_share(picker):
    picker.set_skeleton(*MOUSE)
    _untick(picker, "tail_base")
    picker.set_skeleton(("nose", "tail_base", "tail_tip"), ())
    assert picker.value() == ["nose", "tail_tip"]  # tail_base stays off, the new one on
    picker.set_value(None)
    assert picker.value() is None


def test_the_trail_wording_says_what_the_ticks_are_for(qapp):
    """The clip window's Trail points reuses this picker; its note talks about trails, and a
    split pair means nothing there, so it is not called out."""
    picker = SkeletonPointsPicker(wording="trail")
    assert "No keypoints" in picker._note.text()
    picker.set_skeleton(*MOUSE)
    assert picker._note.text() == "Every point leaves a trail."
    _untick(picker, "left_forepaw")
    assert picker._note.text() == "Only the ticked points leave a trail."
    with pytest.raises(ValueError, match="wording"):
        SkeletonPointsPicker(wording="tracks")
