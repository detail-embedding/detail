"""A processing run is checked against the free space where the experiment is (#743).

**The class.** A run that writes more than the drive holds is not refused anywhere: it starts,
fills the drive, and fails partway, taking whatever else needed that drive with it. On
2026-10-09 a remote embedding batch filled the Windows drive behind a WSL virtual disk to
17 MB, and WSL crashed with the run's coordinator inside it. Exports already refused such a
launch (`Experiment.estimate_export`); processing, the stage family that writes ~190 GB for one
embedding pass of a 157-recording study, did not.

**The rule.** `Experiment.run_processing` and `retry_run` sum every recording's
`ProcessingModule.estimate_output_bytes` (an abstract method: a stage cannot ship without one)
from its validated config, compare it with `disk_space.free_space` (the Windows drive behind a
WSL disk when that is tighter), and raise `InsufficientSpaceError` before anything is recorded,
unless `ignore_free_space`. The estimate and the free space go on the run record, as numbers
only (no absolute path may enter an experiment file).

**The guard**: the refusal, the override and the record, through the real `run_processing` /
`retry_run` with a stand-in stage whose estimate the test sets; the free space is the only
thing faked. The arithmetic of each first-party stage's estimate is pinned below it.
"""

from dataclasses import dataclass

import pytest

from detail.api.experiment import Experiment
from detail.core import disk_space
from detail.core.dataclasses import _JOB_TAGS, _JOB_TYPES
from detail.executor.runners import (
    JOB_RUNNERS,
    PROCESSING_MODULES,
    ProcessingModule,
    import_runner_modules,
    register_processing_module,
)
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.storage import runs as runs_store

_MODULE_ID = "_space_test"
GB = 1024**3


@dataclass
class _SpaceJob:
    job_id: str
    recording_id: str

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "recording_id": self.recording_id}

    @classmethod
    def from_dict(cls, d: dict) -> "_SpaceJob":
        return cls(**d)


class _SpaceModule(ProcessingModule):
    """Writes ``config["bytes"]`` into every recording, by its own estimate."""

    module_id = _MODULE_ID
    id_prefix = "spc"
    runs_key = "_space_runs"
    upstream = "video"
    job_class = _SpaceJob
    progress_event_class = None

    def validate(self, recording_dir, recording_data, config):
        return {**config, "validated": True}

    def build_job(self, recording_id, run_id, config):
        return _SpaceJob(job_id=run_id, recording_id=recording_id)

    def runner(self, job, run_dir, progress_cb, stop_event):
        pass

    def run_dir_for(self, experiment_root, job):
        return experiment_root

    def output_path(self, run_id):
        return f"_space/{run_id}/"

    def estimate_output_bytes(self, recording_dir, recording_data, config):
        assert config.get("validated"), "estimated from the *validated* config"
        return int(config["bytes"])


@pytest.fixture
def space_module():
    register_processing_module(_SpaceModule)
    yield PROCESSING_MODULES[_MODULE_ID]
    PROCESSING_MODULES.pop(_MODULE_ID, None)
    JOB_RUNNERS.pop(_MODULE_ID, None)
    _JOB_TYPES.pop(_MODULE_ID, None)
    _JOB_TAGS.pop(_SpaceJob, None)


@pytest.fixture
def drive(monkeypatch):
    """The free space the check sees, settable per test; the coordinator spawn is stubbed so
    a launch that passes leaves no detached process behind."""
    state = {"free": 100 * GB, "where": "the test drive"}
    monkeypatch.setattr(
        disk_space, "free_space", lambda _path: (state["free"], state["where"])
    )
    monkeypatch.setattr("detail.api.experiment.spawn_coordinator", lambda *a, **k: 4321)
    return state


@pytest.fixture
def proj(tmp_path):
    experiment = Experiment.create("proj", tmp_path)
    for sid in ("recording_001", "recording_002"):
        experiment.create_recording(sid)
    return experiment


SIDS = ["recording_001", "recording_002"]


def test_a_run_that_does_not_fit_is_refused_before_anything_is_recorded(
    proj, space_module, drive
):
    drive["free"] = 20 * GB  # 2 x 10 GB + the 5 GB kept spare does not fit
    with pytest.raises(disk_space.InsufficientSpaceError) as caught:
        proj.run_processing(_MODULE_ID, SIDS, {"bytes": 10 * GB})

    message = str(caught.value)
    assert "20.0 GB" in message and "the test drive" in message
    assert "--ignore-free-space" in message
    assert caught.value.estimate.per_recording == {sid: 10 * GB for sid in SIDS}
    assert proj.processing_runs == [], "refused before a run id was reserved"


def test_the_refusal_is_a_value_error_so_every_surface_reports_it(
    proj, space_module, drive
):
    """The CLI and the GUI catch ``ValueError`` around a launch; a refusal that were not one
    would surface as a crash."""
    drive["free"] = 1 * GB
    with pytest.raises(ValueError):
        proj.run_processing(_MODULE_ID, SIDS, {"bytes": 1})


def test_a_run_that_fits_launches_and_records_its_estimate(proj, space_module, drive):
    run = proj.run_processing(_MODULE_ID, SIDS, {"bytes": 3 * GB})

    record = runs_store.read_run(proj.dir, run.id)
    assert record["space"] == {
        "estimated_output_bytes": 6 * GB,
        "free_bytes_at_launch": 100 * GB,
        "fits": True,
    }


def test_ignore_free_space_launches_and_says_so_on_the_record(
    proj, space_module, drive
):
    drive["free"] = 1 * GB
    run = proj.run_processing(
        _MODULE_ID, SIDS, {"bytes": 10 * GB}, ignore_free_space=True
    )

    space = runs_store.read_run(proj.dir, run.id)["space"]
    assert space["fits"] is False
    assert space["estimated_output_bytes"] == 20 * GB


def test_the_record_holds_numbers_not_where(proj, space_module, drive):
    """``where`` can be an absolute path, which no experiment file may hold."""
    drive["where"] = "/home/someone/experiments"
    run = proj.run_processing(_MODULE_ID, SIDS, {"bytes": 1})
    record_text = (runs_store.run_dir(proj.dir, run.id) / "run.yaml").read_text(
        encoding="utf-8-sig"
    )
    assert "/home/" not in record_text


def test_a_retry_is_checked_against_what_it_will_rewrite(proj, space_module, drive):
    """A retry re-runs each incomplete recording whole, so it is a launch like any other —
    `process retry` was how the run that filled the drive in #743 was restarted."""
    run = proj.run_processing(_MODULE_ID, SIDS, {"bytes": 1 * GB})
    record = runs_store.read_run(proj.dir, run.id)
    record["status"] = "failed"
    record["per_recording"] = {
        "recording_001": {"run_id": "spc_01", "status": "complete"},
        "recording_002": {"run_id": "spc_01", "status": "failed"},
    }
    record["config"]["bytes"] = 30 * GB
    record.pop("coordinator_pid", None)
    runs_store.write_run(proj.dir, run.id, record)

    drive["free"] = 20 * GB
    with pytest.raises(disk_space.InsufficientSpaceError) as caught:
        proj.retry_run(run.id)
    assert caught.value.estimate.per_recording == {"recording_002": 30 * GB}
    assert runs_store.read_run(proj.dir, run.id)["status"] == "failed", "untouched"

    proj.retry_run(run.id, ignore_free_space=True)
    assert runs_store.read_run(proj.dir, run.id)["space"]["fits"] is False


# --------------------------------------------------------------------------- #
# The up-front work reports where it is
# --------------------------------------------------------------------------- #
def _pass(stage: str, names: list[str]) -> list[tuple[str, float, str]]:
    """What one pass over the recordings ``names`` reports: each announced by name before it is
    read (these have no name of their own, so their id), then the end."""
    total = len(names)
    return [
        (stage, index / total, f"{name} ({index + 1} of {total})")
        for index, name in enumerate(names)
    ] + [(stage, 1.0, f"{total} of {total}")]


@pytest.mark.parametrize("selected", [SIDS, SIDS[:1]], ids=["two", "one"])
def test_a_launch_reports_every_recording_through_both_passes(
    proj, space_module, drive, selected
):
    """Validation, then the space estimate, each reads every recording — minutes over a large
    experiment, so each names the recording and says "N of M" rather than leaving one sentence on screen. The single
    recording is the ordinary first launch, and a report that only moves between several
    recordings shows nothing there."""
    events = []
    proj.run_processing(
        _MODULE_ID, selected, {"bytes": 1}, progress=lambda *e: events.append(e)
    )

    assert events == _pass("Validating recordings", selected) + _pass(
        "Estimating disk space", selected
    )


def test_a_launch_refused_for_space_has_already_reported_both_passes(
    proj, space_module, drive
):
    """The refusal comes after the estimate, so the user who sees "does not fit" has watched
    the work that produced it."""
    drive["free"] = 1 * GB
    events = []
    with pytest.raises(disk_space.InsufficientSpaceError):
        proj.run_processing(
            _MODULE_ID, SIDS, {"bytes": 10 * GB}, progress=lambda *e: events.append(e)
        )

    assert {stage for stage, _f, _d in events} == {
        "Validating recordings",
        "Estimating disk space",
    }
    assert events[-1] == ("Estimating disk space", 1.0, "2 of 2")


def test_a_retry_reports_the_recordings_it_will_rerun(proj, space_module, drive):
    run = proj.run_processing(_MODULE_ID, SIDS, {"bytes": 1})
    record = runs_store.read_run(proj.dir, run.id)
    record["status"] = "failed"
    record["per_recording"] = {
        "recording_001": {"run_id": "spc_01", "status": "complete"},
        "recording_002": {"run_id": "spc_01", "status": "failed"},
    }
    record.pop("coordinator_pid", None)
    runs_store.write_run(proj.dir, run.id, record)

    events = []
    proj.retry_run(run.id, progress=lambda *e: events.append(e))

    assert events == _pass("Validating recordings", ["recording_002"]) + _pass(
        "Estimating disk space", ["recording_002"]
    ), "the one incomplete recording, not the two the run holds"


# --------------------------------------------------------------------------- #
# Each first-party stage's estimate
# --------------------------------------------------------------------------- #
def _recording(frames: int) -> dict:
    return {"video": {"files": [{"frame_count": frames}]}}


@pytest.fixture(scope="module")
def modules():
    import_runner_modules(PIPELINE_RUNNERS)
    return PROCESSING_MODULES


def test_embedding_is_rows_times_the_vector(modules):
    est = modules["embedding"].estimate_output_bytes
    cfg = {"model_id": "dinov3_convnext_base", "model_variant": "convnext_base"}
    row = 4 * 1024 + 96

    assert est(None, _recording(1000), {**cfg, "track_ids": [0]}) == 1000 * row
    assert est(None, _recording(1000), {**cfg, "track_ids": [0, 1]}) == 2000 * row
    assert est(None, _recording(1000), {**cfg, "every_n_frames": 2}) == 500 * row
    assert est(None, _recording(1000), {**cfg, "readout": "gap_gmp"}) == 1000 * (
        4 * 2048 + 96
    )
    small = {"model_id": "dinov3_convnext_base", "model_variant": "convnext_small"}
    assert est(None, _recording(1000), small) == 1000 * (4 * 768 + 96)


def test_embedding_matches_what_a_measured_run_wrote(modules):
    """A real 1024-d single-track run of 394,773 frames wrote 1.46 GB (2026-10-09). The
    estimate may sit above it, never far below — an undershoot is the one that fills a
    drive."""
    estimate = modules["embedding"].estimate_output_bytes(
        None,
        _recording(394_773),
        {"model_id": "dinov3_convnext_base", "model_variant": "convnext_base"},
    )
    measured = 1.46 * GB
    assert measured <= estimate <= 1.25 * measured


@pytest.mark.parametrize("module_id", ["detection", "tracking", "keypoint_detect"])
def test_frame_stages_scale_with_frames_and_stride(modules, module_id):
    est = modules[module_id].estimate_output_bytes
    one = est(None, _recording(10_000), {})
    assert one > 0
    assert est(None, _recording(20_000), {}) == 2 * one
    assert est(None, _recording(20_000), {"every_n_frames": 2}) == one


def test_every_stage_estimates_an_empty_recording_without_raising(modules, tmp_path):
    """Registry-driven: a stage added tomorrow is held to it the day it lands."""
    for module_id, module in modules.items():
        value = module.estimate_output_bytes(tmp_path, {}, {})
        assert isinstance(value, int) and value >= 0, module_id
