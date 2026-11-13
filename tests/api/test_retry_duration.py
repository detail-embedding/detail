"""A processing run's duration across a Retry (§15).

**The class: a clock that restarts when its owner does.** Retry spawns a new coordinator
(``Experiment.retry_run``), whose clock starts at zero, and the run's ``duration_s`` was that
clock. An 80-recording detection that had been going for hours read ``0.6`` seconds the moment it
was retried, and the Active-runs table said so. A distill resume had learned to carry its earlier
time over (``_prior_distill_elapsed``); a processing retry had not.

**It hid because no test looked at the duration across a retry**: every retry test checked which
recordings re-ran and what the record rolled up to, and the duration was never an assertion. The
ETA is deliberately *not* the whole run's time (it divides by the work since this coordinator
began, ``_done_at_start``), so only the duration the record reports carries the earlier time.
"""

from detail.api.experiment import Experiment
from detail.storage import runs as runs_store

from tests.api.test_experiment import (
    TRACKING_CONFIG,
    _failed_tracking_run,
    _video_recording,
)


def test_a_retried_run_keeps_the_time_it_had_already_taken(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    run_id = _failed_tracking_run(proj)
    record = runs_store.read_run(proj.dir, run_id)
    record["duration_s"] = 7200.0  # two hours, before the failure
    runs_store.write_run(proj.dir, run_id, record)

    proj.retry_run(run_id).wait()

    after = runs_store.read_run(proj.dir, run_id)
    assert after["status"] == "complete"
    assert after["duration_s"] >= 7200.0  # not reset to the seconds since Retry
    assert after["duration_s"] < 7200.0 + 600  # and it is not counted twice


def test_a_fresh_runs_duration_starts_at_zero(tmp_path):
    """The control: nothing is carried into a run that has no earlier time."""
    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    run = proj.run_processing("tracking", ["recording_001"], dict(TRACKING_CONFIG))
    run.wait()
    assert runs_store.read_run(proj.dir, run.id)["duration_s"] < 600
