"""How much of a recording a track actually holds, and the shape of what it misses.

**Why the shape and not just the total.** `tracks.zarr` records `n_gaps` as
``(last_frame - first_frame + 1) - n_present_frames`` — a count of missing *frames*. A
thousand isolated one-frame losses and one thousand-frame dropout are the same number
there, and they are completely different problems: the first is a per-frame gate rejecting
the subject, the second is the animal leaving or the tracker losing it. Reporting the total
alone tells a user something is wrong and nothing about what.

**Why the span and not the timeline.** Frames before a track appears and after it leaves
are not gaps *in it*. Counting them would report a track covering the back half of a
recording as catastrophically broken, which is a different claim from the true one.
"""

import numpy as np

from detail.api.run_preview import track_coverage


def test_it_counts_gap_runs_not_missing_frames():
    present = np.ones(60, dtype=bool)
    present[20:25] = False  # one run of five
    present[40] = False  # and one singleton

    stats = track_coverage(present, fps=30)
    assert stats["frames_total"] == 60
    assert stats["frames_covered"] == 54
    assert stats["fraction"] == 0.9
    assert stats["n_gaps"] == 2, "six missing frames reported as six gaps"
    assert stats["longest_gap"] == 5
    assert stats["longest_gap_s"] == 5 / 30


def test_the_same_total_with_a_different_shape_reads_differently():
    """The distinction the number in `tracks.zarr` cannot make."""
    scattered = np.ones(60, dtype=bool)
    scattered[[10, 20, 30, 40, 50, 55]] = False
    one_dropout = np.ones(60, dtype=bool)
    one_dropout[20:26] = False

    a, b = track_coverage(scattered), track_coverage(one_dropout)
    assert a["frames_covered"] == b["frames_covered"]  # identical totals
    assert (a["n_gaps"], a["longest_gap"]) == (6, 1)
    assert (b["n_gaps"], b["longest_gap"]) == (1, 6)


def test_frames_outside_the_tracks_span_are_not_its_gaps():
    present = np.zeros(100, dtype=bool)
    present[50:] = True

    stats = track_coverage(present)
    assert stats["n_gaps"] == 0, "the frames before the track began counted as a gap"
    assert stats["first_frame"] == 50 and stats["last_frame"] == 99
    # Coverage is still honest about the half it does not hold.
    assert stats["fraction"] == 0.5


def test_a_track_with_no_frames_at_all():
    stats = track_coverage(np.zeros(10, dtype=bool))
    assert stats["frames_covered"] == 0
    assert stats["fraction"] == 0.0
    assert stats["first_frame"] is None and stats["longest_gap"] == 0


def test_a_perfect_track_reports_no_gaps():
    stats = track_coverage(np.ones(10, dtype=bool), fps=30)
    assert stats["fraction"] == 1.0
    assert stats["n_gaps"] == 0
    assert stats["longest_gap_s"] is None, "a duration for a gap that does not exist"


def test_seconds_are_omitted_rather_than_guessed_without_a_frame_rate():
    present = np.ones(10, dtype=bool)
    present[5] = False
    assert track_coverage(present)["longest_gap_s"] is None
    assert track_coverage(present, fps=10)["longest_gap_s"] == 0.1
