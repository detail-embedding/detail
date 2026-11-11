"""A progress number must describe the work in front of it.

PR #171 fixed one instance of this in the distill fold — nominal stage weights fed into a
time extrapolation, reading minutes left on a run with over an hour to go. This is the sweep for the rest
of the class: a nominal weight, a denominator from a different unit, or a clock measuring a
different span.

Each test asserts the *number a user reads*, not that a call returned.
"""

import pytest

from detail.core.dataclasses import TrainingProgressEvent
from detail.core.progress import remaining_seconds


class _Clock:
    """A scripted monotonic clock, so an ETA assertion is exact, not timing-dependent."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


# --------------------------------------------------------------------------- #
# a phase's ETA comes from the running phase, not the weighted bar
# --------------------------------------------------------------------------- #
def test_a_phase_eta_measures_the_phase_not_the_run():
    """The removed clusterer fit's `_global_fraction` weighted load->0.3 and DTW 0.3->0.9.
    Dividing the whole-fit clock by that put the ETA at ~2.33x the *load* time the moment the
    hours-long DTW began."""
    from detail.core.progress import PhaseClock

    clock = PhaseClock(now=0.0)
    clock.eta("load", 0.5, now=30.0)  # halfway through a one-minute load
    at_handoff = clock.eta("dtw", 0.0, now=60.0)  # DTW starts: nothing measurable yet
    assert at_handoff is None  # ...and emphatically not 2.33 x 60 s

    after_5_min = clock.eta("dtw", 0.1, now=60.0 + 300.0)
    # 300 s bought 10% of the DTW, so ~2700 s remain. The old code divided the whole clock
    # (360 s) by the weighted fraction (0.36) and reported ~640 s.
    assert after_5_min == pytest.approx(2700.0, rel=0.02)
    assert after_5_min > 2000
    assert after_5_min != pytest.approx(remaining_seconds(0.36, 360.0), rel=0.05)


def test_the_phase_clock_restarts_on_every_phase_change():
    from detail.core.progress import PhaseClock

    clock = PhaseClock(now=0.0)
    clock.eta("load", 0.9, now=100.0)  # 100 s spent in `load`

    assert (
        clock.eta("dtw", 0.0, now=100.0) is None
    )  # dtw's first tick: nothing measured
    # ...and its second tick measures dtw's 10 s, never load's 100
    assert clock.eta("dtw", 0.5, now=110.0) == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# a resumed training run's ETA divides by the epochs *this* process ran
# --------------------------------------------------------------------------- #
def test_resumed_training_eta_counts_only_this_process_s_epochs():
    """The epoch twin of ProcessingProgress.recordings_at_start. Resuming at epoch 40 of
    100 with 10 epochs done here must not infer the throughput of all 40."""
    epochs_at_start = []

    def eta(epoch: int, total_epochs: int, elapsed: float):
        if not epochs_at_start:
            epochs_at_start.append(epoch - 1)
        done_here = epoch - epochs_at_start[0]
        total_here = total_epochs - epochs_at_start[0]
        return (
            remaining_seconds(done_here / total_here, elapsed) if total_here else None
        )

    eta(41, 100, 60.0)  # first tick after resuming at epoch 40
    after_10 = eta(50, 100, 600.0)  # 10 epochs in 600 s → 60 s/epoch, 50 left

    assert after_10 == pytest.approx(50 * 60.0, rel=0.02)
    # the run-wide fraction would have said 600·(1−0.5)/0.5 = 600 s, 5x low
    assert after_10 != pytest.approx(remaining_seconds(0.5, 600.0))


# --------------------------------------------------------------------------- #
# the bar must not read 100% while work continues
# --------------------------------------------------------------------------- #
def test_the_training_preview_tail_does_not_report_a_finished_run():
    """The tail step emitted epoch=1/total=1, and the renderer maps epoch/total_epochs to
    the bar — so it read 100% while the preview was still being written."""
    from detail.gui.workers import fit_progress_fraction

    tail = TrainingProgressEvent(
        epoch=100,
        total_epochs=101,
        metrics={},
        elapsed_s=0.0,
        eta_s=None,
        message="writing preview",
    )
    fraction, status = fit_progress_fraction(tail)
    assert fraction < 1.0
    assert "writing preview" in status
