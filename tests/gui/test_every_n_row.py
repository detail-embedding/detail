"""The "Use every N frames" row — the one place a run's cadence is chosen and shown.

Two things it has to get right: the rate is on screen **always** (not only when the run is
thinned, which is when the user least needs telling), and downstream it cannot be set to a
value the stage above never produced.
"""

import pytest

pytest.importorskip("PyQt6")

from detail.configs.detection import DetectionConfig  # noqa: E402
from detail.configs.embedding import EmbeddingConfig  # noqa: E402
from detail.gui.modules._every_n import (  # noqa: E402
    EveryNRow,
    describe,
    format_rate,
    source_rate,
)


class _Rec:
    def __init__(self, fps):
        self.video_fps = fps


# --------------------------------------------------------------------------- #
# The line beside the number
# --------------------------------------------------------------------------- #
def test_the_rate_is_shown_even_when_nothing_is_skipped():
    """The common case. A blank line here is what the old note did, and it left the
    user with no idea what rate they were about to run at."""
    assert describe(1, 1, (30.0, 30.0)) == "30 fps (source 30 fps)"


def test_the_rate_falls_with_the_number():
    assert describe(4, 1, (30.0, 30.0)) == "7.5 fps (source 30 fps)"
    assert describe(8, 1, (60.0, 60.0)) == "7.5 fps (source 60 fps)"


def test_a_selection_spanning_frame_rates_shows_the_range():
    """One recording's rate silently standing for all of them is the quiet wrongness
    this label exists to prevent."""
    assert describe(4, 1, (30.0, 60.0)) == "7.5–15 fps (sources vary)"


def test_without_a_known_rate_it_still_says_what_it_will_do():
    assert describe(4, 1, None) == "every 4th frame"
    assert describe(1, 1, None) == "every frame"


def test_downstream_says_what_share_of_its_input_it_keeps():
    """``every 4th frame`` under a detection run at 4 is *all* of them, not a quarter —
    the number alone cannot say which, so the clause does."""
    assert describe(4, 4, (30.0, 30.0)).endswith(" · every detected frame")
    assert describe(8, 4, (30.0, 30.0)).endswith(" · half the detected frames")
    assert describe(12, 4, (30.0, 30.0)).endswith(" · a third of the detected frames")
    assert describe(20, 4, (30.0, 30.0)).endswith(" · one in 5 detected frames")


def test_rates_round_to_something_readable():
    assert format_rate(29.97) == "30"  # a hair off whole reads as whole
    assert format_rate(7.4925) == "7.5"
    assert format_rate(15.0) == "15"


def test_source_rate_ignores_recordings_with_no_video():
    assert source_rate([_Rec(30.0), _Rec(None)]) == (30.0, 30.0)
    assert source_rate([_Rec(None)]) is None
    assert source_rate([]) is None


# --------------------------------------------------------------------------- #
# The widget
# --------------------------------------------------------------------------- #
def test_detection_row_starts_at_every_frame_and_shows_the_rate(qapp):
    row = EveryNRow(DetectionConfig)
    row.set_recordings([_Rec(30.0)])
    assert row.value() == 1
    assert row._note.text() == "30 fps (source 30 fps)"


def test_downstream_row_cannot_be_clicked_below_its_input(qapp):
    """The floor and the step are the upstream cadence, so the arrows can only ever
    land on a reachable value — asking for frames that were never processed is not a
    state the mouse can reach."""
    row = EveryNRow(EmbeddingConfig, upstream_noun="tracked")
    row.set_recordings([_Rec(30.0)])
    row.set_upstream(4)

    assert row.value() == 4  # pre-filled with its input's cadence
    assert row._spin.minimum() == 4
    assert row._spin.singleStep() == 4
    row._spin.stepBy(-1)
    assert row.value() == 4  # clamped, not 3
    row._spin.stepBy(1)
    assert row.value() == 8


def test_a_typed_non_multiple_is_caught_and_explained(qapp):
    row = EveryNRow(EmbeddingConfig, upstream_noun="tracked")
    row.set_recordings([_Rec(30.0)])
    row.set_upstream(4)
    row._spin.setValue(6)  # only reachable by typing

    assert "use 4, 8, 12" in row.warning()
    assert row._note.text().startswith("⚠")


def test_switching_upstream_refills_a_now_impossible_value(qapp):
    """Pick a detection run at 4, set 8, then switch to one at 3: 8 is no longer
    reachable, so the row resets rather than sitting on a value that would be refused.
    """
    row = EveryNRow(EmbeddingConfig, upstream_noun="tracked")
    row.set_upstream(4)
    row._spin.setValue(8)
    row.set_upstream(3)
    assert row.value() == 3
    assert row.warning() is None


def test_the_share_clause_names_the_stage_above(qapp):
    row = EveryNRow(EmbeddingConfig, upstream_noun="tracked")
    row.set_recordings([_Rec(30.0)])
    row.set_upstream(2)
    assert row._note.text().endswith(" · every tracked frame")
