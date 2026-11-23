"""Unit tests for the shared progress throttle (core/progress.py)."""

import pytest

from detail.core import progress
from detail.core.progress import (
    ThrottledProgress,
    format_duration,
    remaining_seconds,
)


def test_first_call_always_forwards():
    events = []
    tp = ThrottledProgress(events.append, interval_s=1000.0)
    tp("a")  # first call forwards immediately despite the huge interval
    assert events == ["a"]


def test_throttles_within_interval():
    events = []
    tp = ThrottledProgress(events.append, interval_s=1000.0)
    tp("a")  # forwarded (first)
    tp("b")  # suppressed (within interval)
    tp("c")  # suppressed
    assert events == ["a"]


def test_force_bypasses_throttle():
    events = []
    tp = ThrottledProgress(events.append, interval_s=1000.0)
    tp("a")  # first
    tp("b")  # suppressed
    tp("c", force=True)  # forced through despite the interval
    assert events == ["a", "c"]


def test_none_cb_is_noop():
    tp = ThrottledProgress(None)
    tp("a")
    tp("b", force=True)  # no error, nothing forwarded


def test_interval_none_reads_module_global(monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", 0.0)
    events = []
    tp = ThrottledProgress(events.append)  # interval_s=None -> reads the global
    tp("a")
    tp("b")
    tp("c")
    assert events == ["a", "b", "c"]  # 0.0 interval -> every call forwards


# --- ETA helpers ----------------------------------------------------------- #
def test_remaining_seconds_linear_estimate():
    # Half done after 10s → 10s to go (elapsed * (1 - f) / f).
    assert remaining_seconds(0.5, 10.0) == pytest.approx(10.0)
    assert remaining_seconds(0.25, 30.0) == pytest.approx(90.0)


def test_remaining_seconds_none_when_uncomputable():
    assert remaining_seconds(0.0, 10.0) is None  # no progress yet
    assert remaining_seconds(1.0, 10.0) is None  # already complete
    assert remaining_seconds(1.5, 10.0) is None  # past complete (clamp guard)
    assert remaining_seconds(0.5, 0.0) is None  # no elapsed time


def test_format_duration_units():
    assert format_duration(0) == "0s"
    assert format_duration(45) == "45s"
    assert format_duration(150) == "2m30s"
    assert format_duration(3900) == "1h05m"


# --------------------------------------------------------------------------- #
# message_phase: which declared phase a runner's message names (§16, issue #393)
# --------------------------------------------------------------------------- #

_SCORED_RUN = ("loading", "querying", "scoring")
_SYLLABLES = (
    "loading embeddings",
    "detecting transitions",
    "comparing bouts",
    "clustering",
)


@pytest.mark.parametrize(
    "message, phases, expected",
    [
        # a runner may forward an inner step into its note: the removed query eval's querying
        # phase carried its search's own `loading <subject>` there
        ("querying · Rearing · fold 1/3 · loading rec_001 · 4 matches", _SCORED_RUN, 1),
        (
            "querying · Rearing · fold 1/3 · searching rec_001 · 4 matches",
            _SCORED_RUN,
            1,
        ),
        # a user's behavior / recording name is note text too
        ("scoring · Unloading · aggregating 3 fold(s)", _SCORED_RUN, 2),
        ("clustering · recording_loading_bay", ("loading", "clustering"), 1),
        # a count inside the head is still the phase
        ("comparing 4,182 bouts", _SYLLABLES, 2),
        ("detecting transitions · 30 fps → window 12 rows", _SYLLABLES, 1),
        ("loading rec_001", ("loading", "searching"), 0),
        ("writing metrics", ("scoring", "writing metrics"), 1),
    ],
)
def test_message_phase_reads_the_head_not_the_note(message, phases, expected):
    """The note is the runner's to fill with anything, so a phase word there is not a phase."""
    from detail.core.progress import message_phase

    assert message_phase(message, phases) == expected


def test_message_phase_needs_every_word_of_a_multi_word_phase():
    from detail.core.progress import message_phase

    assert message_phase("comparing · 40 rows", _SYLLABLES) is None


def test_message_phase_places_a_done_message_past_every_phase():
    from detail.core.progress import message_phase

    assert message_phase("done", _SCORED_RUN) == len(_SCORED_RUN)
    assert message_phase("done · 2 clip(s) held back", _SCORED_RUN) == len(_SCORED_RUN)


@pytest.mark.parametrize("message", ["", "starting", "Rearing · loading"])
def test_message_phase_is_none_for_a_message_naming_no_phase(message):
    from detail.core.progress import message_phase

    assert message_phase(message, _SCORED_RUN) is None
