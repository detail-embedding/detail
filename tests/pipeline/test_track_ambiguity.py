"""A bare integer track id stops being a subject the moment a recording has two of them.

Track ids are assigned in order of first appearance *within one tracking run* (§13), so an
integer names a position, not an animal. With one track per recording — every recording in this
project today — that position is unambiguous and the int finds the one subject. With several,
``track_id=0`` is "whichever the tracker numbered first", and that can be a different animal in
each recording.

The modules guarded here **pool subjects across recordings**: query-eval scores retrieval
across every eligible one (a clusterer fit and syllable discovery did too, before they were
removed). Passing an integer that means a different
animal in each would fit or score across animals, and nothing about the output would look
wrong — every recording contributes, the counts are plausible, the number is a number.

Worth being precise about what this is *not*: whether two recordings hold the same animal is a
recording-level fact and is already answered by the §11 grouping variables (``metadata.csv``'s
animal id column). This is only about which subject *within* a video, which is what
``subject_labels`` and ``core.utils.resolve_track_id`` are for — designed in DESIGN_PLAN §13 and
deliberately not built, since nothing yet needs them. This guard is what will tell someone they
do.
"""

import pytest

from detail.core.utils import check_track_is_unambiguous


def test_one_track_is_unambiguous():
    """The case that must not change: every recording that exists today."""
    check_track_is_unambiguous(0, "rec_a", [0], run_id="emb_01")


def test_no_tracks_is_left_to_the_caller():
    """An absent track is a different error, raised by the caller's own membership check."""
    check_track_is_unambiguous(0, "rec_a", [], run_id="emb_01")


def test_two_tracks_raises():
    with pytest.raises(ValueError) as excinfo:
        check_track_is_unambiguous(0, "rec_a", [0, 1], run_id="emb_01")
    message = str(excinfo.value)

    # Whoever reads this is meeting the problem for the first time, so the message has to
    # carry the specifics *and* the reason — not just "ambiguous".
    assert "rec_a" in message, "must name the recording"
    assert "[0, 1]" in message, "must list the tracks it actually found"
    assert "first appearance" in message, "must say why an int is not stable"
    assert "subject_labels" in message, "must point at the designed way out"


def test_the_message_survives_unsorted_input():
    with pytest.raises(ValueError, match=r"\[0, 2, 5\]"):
        check_track_is_unambiguous(0, "rec_a", [5, 0, 2], run_id="emb_01")


@pytest.mark.parametrize(
    "module_path,function",
    [
        ("detail.pipeline.query_eval", "validate"),
    ],
)
def test_every_pooling_module_calls_the_check(module_path, function):
    """Source scan, so a fourth module that pools subjects is not silently exempt.

    Behavioural coverage would need a two-track fixture per module — three full experiments with
    embedding runs — to assert one `if`. The thing that actually goes wrong is a *new* pooling
    module forgetting the call, and that is what this catches.
    """
    import importlib
    import inspect

    module = importlib.import_module(module_path)
    target = getattr(module, function, None)
    source = (
        inspect.getsource(module)
        if target is None
        else inspect.getsource(inspect.getmodule(target))
    )
    assert "check_track_is_unambiguous" in source, (
        f"{module_path} pools subjects across recordings but never checks that its track_id "
        "is unambiguous; an integer means a different animal in each recording once one of "
        "them has more than one track"
    )
