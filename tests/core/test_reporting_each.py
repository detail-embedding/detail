"""``reporting_each``: the one way a synchronous loop over recordings says where it is.

**The class.** Pre-run validation reads every selected recording before a processing run starts
anything, and for a 71-recording experiment (or a network share) that is minutes in which the
launch dialog's status sentence never changed. Nothing was wrong at the line that read the
recordings; the defect was that the loop could not be told apart from a hang.

**The rule.** Each item is announced *before* it is processed (``N of M``, the fraction being the
items already done), and the end is announced at ``1.0``. A one-item loop therefore still moves
the bar at both ends — the shape every earlier cadence bug hid in is a test that only passes with
several items, so the single-item case is asserted on its own.
"""

import pytest

from detail.core.progress import ESTIMATING_STAGE, VALIDATING_STAGE, reporting_each


def _collect(items, **kwargs):
    events = []
    yielded = list(
        reporting_each(items, "Stage", lambda *event: events.append(event), **kwargs)
    )
    return yielded, events


def test_each_item_is_announced_before_it_is_handled_and_the_end_at_one():
    yielded, events = _collect(["a", "b", "c", "d"])

    assert yielded == ["a", "b", "c", "d"]
    assert events == [
        ("Stage", 0.0, "1 of 4"),
        ("Stage", 0.25, "2 of 4"),
        ("Stage", 0.5, "3 of 4"),
        ("Stage", 0.75, "4 of 4"),
        ("Stage", 1.0, "4 of 4"),
    ]


def test_a_single_item_still_reports_at_both_ends():
    """The case a multi-item fixture hides: with one recording the bar must not sit at nothing
    and then vanish."""
    _yielded, events = _collect(["only"])

    assert events == [("Stage", 0.0, "1 of 1"), ("Stage", 1.0, "1 of 1")]


def test_a_label_names_the_item_in_the_detail():
    _yielded, events = _collect(["recording_007"], label=str.upper)

    assert events[0] == ("Stage", 0.0, "RECORDING_007 (1 of 1)")


def test_the_item_is_announced_while_the_loop_body_has_not_run():
    """Before, not after: an announcement made after the item is handled would name the
    recording that just finished while the slow one is the one on screen."""
    seen = []
    handled = []
    for item in reporting_each(["a", "b"], "Stage", lambda *event: seen.append(event)):
        handled.append(item)
        assert len(seen) == len(handled), "the item's own announcement is already out"


def test_no_callback_reports_nothing_and_still_yields():
    assert list(reporting_each(["a", "b"], "Stage", None)) == ["a", "b"]


def test_an_empty_loop_reports_nothing_rather_than_a_finished_bar():
    _yielded, events = _collect([])

    assert events == []


def test_a_loop_that_stops_early_does_not_claim_to_be_finished():
    events = []
    with pytest.raises(RuntimeError):
        for item in reporting_each(
            ["a", "b", "c"], "Stage", lambda *e: events.append(e)
        ):
            if item == "b":
                raise RuntimeError("recording b does not validate")

    assert [e[1] for e in events] == [0.0, pytest.approx(1 / 3)]
    assert all(e[1] < 1.0 for e in events)


def test_the_two_stage_names_the_surfaces_render_are_distinct():
    assert VALIDATING_STAGE != ESTIMATING_STAGE
