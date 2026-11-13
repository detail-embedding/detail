"""A run never starts inside a deleted run's leftovers.

**The class: an identity that outlives its record.** A per-recording run id is a counter
(``api/recording.py::_next_run_id``): max + 1 over the ids ``recording.yaml`` still lists. Deleting
a run removes its entry, so its id is handed out again — but its output is not reliably gone. Delete
is local and the cluster keeps its copy; the upload never overwrites a run directory the cluster
already has (``copy_to_remote``'s ``never_overwrite``), on the premise that this job's own is not
there yet; and copy-back brings the cluster's copy home, original timestamps and all.

**How it showed.** On tillicum, a 12-keypoint ``kp_03`` ran inside a deleted 4-keypoint ``kp_03``
and refused its zarr (``column 'xy' … has trailing shape (4, 2), expected (12, 2)``). A ``kp_01``
on another recording inherited a deleted run's ``cancel`` sentinel and stopped in a second, recorded
``partial``. The quiet case is worse: a store of the same width is opened, verified, and appended to
— the keypoints writer does not skip frames already on disk — so two models' rows share one track
with no error anywhere.

**The fix, both halves.** Locally, ``_next_run_id`` steps past an id whose output directory still
exists (this module). On the cluster, the shepherd reads the ``job.json`` already under this run's
directory before uploading and removes the directory when it names a different ``job_id`` /
``created_at``; one naming this job — a resume, a restarted shepherd — is kept, and one with nothing
readable is left as it always was (``remote_shepherd._clear_foreign_run_dir``, guarded by
``tests/executor/test_remote_shepherd.py::
test_a_cluster_run_dir_left_by_another_job_is_replaced_before_the_upload``).

**Where it stops.** Ids are still reused when nothing is left on either side, which is harmless by
construction. A persisted high-water mark would retire reuse entirely, at the price of a
``recording.yaml`` field; it was not needed to make the reuse safe.
"""

from detail.api.experiment import Experiment
from detail.core.dataclasses import JobStatus
from tests.api.test_experiment import TRACKING_CONFIG, _video_recording


def test_a_run_never_takes_an_id_whose_output_is_still_on_disk(tmp_path, monkeypatch):
    class _Executor:
        def submit(self, job):
            self.job = job

        def status(self, run_id):
            return JobStatus.PENDING

    executor = _Executor()
    monkeypatch.setattr("detail.api.recording.executor_for", lambda *a: executor)
    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    recording = proj.get_recording("recording_001")
    (recording.dir / "tracking" / "track_01").mkdir(parents=True)  # no record names it

    run = recording.run("tracking", dict(TRACKING_CONFIG))

    assert run.id == executor.job.job_id == "track_02"
    assert [r["id"] for r in recording.tracking_runs] == ["track_02"]
