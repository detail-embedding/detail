"""``Experiment.run_analysis`` sends a remote-capable module's job to its remote target (§6
"Remote analysis"), and refuses a remote ``compute`` for every other module.

**The class: a compute target that a validate() can drop.** An analysis module's ``validate()``
returns its own normalized config, and most rebuild it from their pydantic model, which knows
nothing of ``compute``. Routing on the *validated* config alone would send a remote request for
such a module to this machine without a word. So the refusal reads the config as given, before
any module sees it, and a remote-capable module that loses the target in its own ``validate()``
is refused too rather than run here.
"""

import dataclasses

import pytest

from detail.api.experiment import Experiment
from detail.core.dataclasses import ComputeTarget, JobStatus
from detail.executor import remote as remote_module
from detail.executor.runners import ANALYSIS_MODULES, AnalysisModule

REMOTE = ComputeTarget("remote", None, None, None, remote_target="tillicum")


@dataclasses.dataclass
class _Job:
    job_id: str
    compute: dict | None = None


class _Module(AnalysisModule):
    module_id = "remote_stub"
    id_prefix = "an"
    requires: list[str] = []
    job_class = _Job
    phases = ("running",)
    remote_capable = True
    keeps_compute = True

    def validate(self, experiment_dir, config):
        return dict(config) if self.keeps_compute else {"source_recording_ids": []}

    def build_job(self, run_id, config):
        return _Job(run_id, config.get("compute"))

    def runner(self, job, run_dir, progress_cb, stop_event):
        raise NotImplementedError  # these tests stop at submit()

    def run_dir_for(self, experiment_root, job):
        return experiment_root / "analysis" / job.job_id

    def output_manifest(self, run_id, config):
        return []


class _LocalOnly(_Module):
    module_id = "local_stub"
    remote_capable = False


class _Forgetful(_Module):
    module_id = "forgetful_stub"
    keeps_compute = False


class _FakeExecutor:
    made: list = []

    def __init__(self, experiment_root, *args, runner_modules=(), **kwargs):
        type(self).made.append((type(self).__name__, args))

    def submit(self, job):
        return job.job_id

    def status(self, job_id):
        return JobStatus.RUNNING


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    for module in (_Module(), _LocalOnly(), _Forgetful()):
        monkeypatch.setitem(ANALYSIS_MODULES, module.module_id, module)
    _FakeExecutor.made = []
    monkeypatch.setattr(remote_module, "LocalExecutor", _FakeExecutor)
    monkeypatch.setattr(remote_module, "RemoteExecutor", _FakeExecutor)
    return Experiment.create("p", tmp_path)


def test_a_remote_capable_module_is_submitted_to_its_target(experiment):
    run = experiment.run_analysis("remote_stub", {"compute": REMOTE})

    assert _FakeExecutor.made == [("_FakeExecutor", ("tillicum",))]
    record = experiment.analysis_record(run.id)
    assert record["compute"]["remote_target"] == "tillicum"


def test_a_module_without_compute_runs_here(experiment):
    experiment.run_analysis("remote_stub", {})

    assert _FakeExecutor.made == [("_FakeExecutor", ())]


@pytest.mark.parametrize("compute", [REMOTE, REMOTE.to_dict()])
def test_a_module_that_cannot_run_remotely_is_refused_before_it_validates(
    experiment, compute
):
    with pytest.raises(ValueError, match="local_stub cannot run on a remote target"):
        experiment.run_analysis("local_stub", {"compute": compute})
    assert not _FakeExecutor.made
    assert not experiment.analysis_runs


def test_a_validate_that_drops_the_target_is_refused_not_run_here(experiment):
    with pytest.raises(ValueError, match="dropped its remote compute target"):
        experiment.run_analysis("forgetful_stub", {"compute": REMOTE})
    assert not _FakeExecutor.made
