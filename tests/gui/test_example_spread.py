"""The spread hint beside Mark examples (detail/gui/example_spread.py): which gap it names, in
which order, and what it refuses to call a gap."""

from detail.gui.example_spread import missing_values, spread_hint, spread_text

_NAMES = {"r1": "rec 1", "r2": "rec 2", "r3": "rec 3"}


def _clip(recording_id, track=0):
    return {"recording_id": recording_id, "track_id": track}


def test_no_examples_says_so():
    assert spread_hint("attack", [], _NAMES) == "attack: no examples yet."
    assert spread_text([]) == "—"


def test_all_from_one_recording_is_named_first():
    hint = spread_hint("attack", [_clip("r1"), _clip("r1", 1)], _NAMES)
    assert (
        hint == "attack: all 2 examples are from rec 1. Add one from another recording."
    )


def test_one_recording_searched_is_not_a_gap():
    hint = spread_hint("attack", [_clip("r1"), _clip("r1", 1)], {"r1": "rec 1"})
    assert "another recording" not in hint and "2 animals" in hint


def test_all_of_one_animal_is_named_when_only_one_recording_is_searched():
    hint = spread_hint("attack", [_clip("r1"), _clip("r1")], {"r1": "rec 1"})
    assert "one animal (track 0 in rec 1)" in hint


def test_a_balance_value_with_no_examples_is_named():
    metadata = {
        "r1": {"treatment": "saline"},
        "r2": {"treatment": "CNO"},
        "r3": {"treatment": ""},
    }
    clips = [_clip("r1"), _clip("r3")]
    assert missing_values(clips, _NAMES, metadata, "treatment") == ["CNO"]
    hint = spread_hint("attack", clips, _NAMES, metadata=metadata, column="treatment")
    assert hint == "attack: no examples from treatment = CNO."


def test_an_empty_metadata_cell_is_not_a_value_to_sample():
    metadata = {"r1": {"treatment": "saline"}, "r2": {"treatment": " "}}
    clips = [_clip("r1"), _clip("r3")]
    assert missing_values(clips, _NAMES, metadata, "treatment") == []


def test_a_good_spread_says_what_it_covers():
    metadata = {"r1": {"sex": "f"}, "r2": {"sex": "m"}}
    clips = [_clip("r1"), _clip("r2")]
    hint = spread_hint("attack", clips, _NAMES, metadata=metadata, column="sex")
    assert hint == "attack: 2 examples from 2 recordings and 2 animals, every sex."
    assert spread_text(clips) == "2 recs · 2 animals"


def test_one_example_reads_as_one():
    hint = spread_hint("hunch", [_clip("r1")], _NAMES)
    assert (
        hint == "hunch: the only example is from rec 1. Add one from another recording."
    )
