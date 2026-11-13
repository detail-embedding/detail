"""Camera view resolution (§10): the recording's own, else the experiment default, else not set.

The class this guards is a missing fact turning into a wrong one. A view that resolves to
"top down" when nothing was set makes every side-only measure silently meaningless, so
"not set" must stay an answer of its own, and a malformed stored value must raise rather
than read as absent.
"""

import pytest

from detail.core import camera_view


def test_the_recording_own_view_wins_over_the_default():
    resolved = camera_view.resolve({"camera_view": "top_down"}, {"camera_view": "side"})
    assert (resolved.view, resolved.source) == ("top_down", "recording")


def test_all_top_down_needs_a_view_on_every_one():
    """Not set is not top down: one unset view, or none at all, is never "all top down"."""
    top, side = camera_view.CameraView("top_down"), camera_view.CameraView("side")
    unset = camera_view.CameraView(None)
    assert camera_view.all_top_down([top, top])
    assert not camera_view.all_top_down([top, side])
    assert not camera_view.all_top_down([top, unset])
    assert not camera_view.all_top_down([])


def test_a_recording_without_a_view_inherits_the_default():
    resolved = camera_view.resolve({}, {"camera_view": "side"})
    assert (resolved.view, resolved.source) == ("side", "experiment")
    assert resolved.described == "Side view (experiment default)"


def test_nothing_set_anywhere_is_not_set_never_a_guess():
    resolved = camera_view.resolve({}, {})
    assert resolved.view is None and resolved.source is None
    assert not resolved.is_set
    assert resolved.label == "Not set"


def test_other_carries_its_note_into_the_label():
    resolved = camera_view.resolve(
        {"camera_view": "other", "camera_view_note": "45° from above"}, {}
    )
    assert resolved.label == "Other: 45° from above"
    assert resolved.to_record() == {
        "view": "other",
        "note": "45° from above",
        "source": "recording",
    }


@pytest.mark.parametrize(
    "view, note, fragment",
    [
        ("sideways", None, "unknown camera view 'sideways'"),
        ("other", None, "needs a description"),
        ("other", "   ", "needs a description"),
        ("side", "a note", "only kept for 'other'"),
        (None, "a note", "needs a view of 'other'"),
    ],
)
def test_validate_refuses_what_it_cannot_store(view, note, fragment):
    with pytest.raises(ValueError, match=fragment):
        camera_view.validate(view, note)


def test_a_malformed_stored_value_raises_rather_than_reading_as_absent():
    with pytest.raises(ValueError, match="unknown camera view"):
        camera_view.resolve({"camera_view": "diagonal"}, {"camera_view": "side"})


def test_write_sets_both_keys_and_clearing_removes_both():
    block = {"id": "r"}
    camera_view.write(block, "other", "  mirror behind  ")
    assert block == {
        "id": "r",
        "camera_view": "other",
        "camera_view_note": "mirror behind",
    }
    camera_view.write(block, "side", None)
    assert block == {"id": "r", "camera_view": "side"}
    camera_view.write(block, None, None)
    assert block == {"id": "r"}


def test_every_view_has_a_label_and_other_is_last():
    assert list(camera_view.VIEWS) == ["top_down", "side", "angled_down", "other"]
    assert all(camera_view.VIEWS.values())
