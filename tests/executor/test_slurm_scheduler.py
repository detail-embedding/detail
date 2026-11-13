"""SlurmScheduler — sbatch rendering + squeue/sacct parsing, all against a mocked
ssh_transport (no real cluster needed).

**The class guarded by the `unreachable` tests: being unable to ask recorded as an answer.**
`poll_state` read "squeue and sacct both came back empty" as "the scheduler has no record of this
job, so it failed". An ssh that never connected (a laptop that slept, an expired session, a
network change) is empty too, so a healthy multi-hour job was recorded `failed`. The two ways of
being empty are told apart by who produced the exit status: ssh's own 255 means the question never
ran, where squeue exiting 1 is Slurm saying the job is not queued. The first is `UNREACHABLE`, a
non-terminal state the shepherd waits on; the second stays `FAILED`, and its control test below
must keep passing or a job that is really gone would be waited on for a week. `cancel` had the
same shape (a `scancel` into a dead connection read as sent) and now says whether it was
delivered. The shepherd side, and how the run behaves while waiting, is written up in
`test_remote_shepherd.py`.

**The class guarded by the `sacct` state tests: a lookup whose default is a verdict, missing the
values it was never expected to see.** sacct was written as "the historical record, once the job
has left squeue", so its map listed only end states and defaulted the rest to `FAILED`. But sacct
is asked whenever squeue gives no state, and a controller too busy to answer (squeue exits 1) does
that to a running job; sacct then answers `RUNNING`, which the default read as failed. Embedding
run_011's recording_083 (Slurm job 351119) was recorded failed 30 s in, as "remote job ended
(failed) without a terminal status", while Slurm ran it to `COMPLETED 0:0` 12 minutes later and
its results sat on the cluster. Every state sacct documents is now mapped, and the guard is over
that list, so a state is never left to the default by omission.
"""

import pytest

from detail.configs.remote import RemoteTargetConfig
from detail.executor import ssh_transport
from detail.executor.schedulers import slurm
from detail.executor.schedulers.base import TERMINAL_STATES
from detail.executor.schedulers.base import JobState
from detail.executor.schedulers.slurm import (
    SlurmScheduler,
    activation_prefix,
    render_sbatch_script,
)


def _target(**overrides):
    fields = dict(
        name="tillicum",
        host="tillicum",
        remote_workdir="/gscratch/scrubbed/uwnetid/detail_runs",
    )
    fields.update(overrides)
    return RemoteTargetConfig(**fields)


class _FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_render_sbatch_script_includes_resources_and_worker_invocation():
    target = _target(
        partition="gpu-a40",
        account="mylab",
        conda_env="detail-env",
        modules=["cuda/12.4", "gcc/13.2"],
        gpu_type="h200",
        gpus_per_job=2,
        cpus_per_task=8,
        mem_gb=64,
        walltime="1-00:00:00",
    )
    script = render_sbatch_script(
        target, "det_01", "/scratch/exp/runs/det_01", "detail.pipeline.detection"
    )
    assert script.startswith("#!/bin/bash\n")
    assert "#SBATCH --partition=gpu-a40" in script
    assert "#SBATCH --account=mylab" in script
    assert "#SBATCH --gres=gpu:h200:2" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --time=1-00:00:00" in script
    assert "module load cuda/12.4" in script
    assert "module load gcc/13.2" in script
    assert "conda activate detail-env" in script
    assert "python -P -m detail.executor.worker /scratch/exp/runs/det_01" in script


def test_render_sbatch_script_quotes_an_output_path_containing_a_space():
    """remote_run_dir mirrors the local experiment's own path — including a name a
    user typed, e.g. "Open Field B" — verbatim onto the cluster. An unquoted space
    there splits the `#SBATCH --output=` directive into a second, invalid directive
    line; found live submitting a real job against that exact experiment."""
    target = _target()
    script = render_sbatch_script(
        target,
        "distill_003",
        "/scratch/Open Field B/datasets/dataset_004",
        "detail.pipeline.labeling",
    )
    lines = script.splitlines()
    output_line = next(line for line in lines if line.startswith("#SBATCH --output="))
    assert (
        output_line
        == '#SBATCH --output="/scratch/Open Field B/datasets/dataset_004/slurm-%j.out"'
    )
    # Exactly one directive line was produced — a naive split-on-space would have
    # turned the trailing "BMAD/..." fragment into a second, bogus #SBATCH line.
    assert sum(1 for line in lines if line.startswith("#SBATCH --output")) == 1


def test_render_sbatch_script_omits_unset_optional_directives():
    target = _target()  # no partition/account/gpu_type/conda_env/modules
    script = render_sbatch_script(target, "det_01", "/scratch/exp/runs/det_01", "")
    assert "--partition" not in script
    assert "--account" not in script
    assert "conda activate" not in script
    assert "module load" not in script
    assert "#SBATCH --gres=gpu:1" in script  # no gpu_type -> bare gpu:<n>


def test_render_sbatch_script_exports_runner_modules_before_the_worker_runs():
    """A fresh compute-node shell inherits none of the shepherd's own environment, so
    the worker's job-type registry is empty unless this script exports it — the class
    of bug behind "no runner registered for job type 'detection'" on a real cluster
    (every job type, not just detection: the export was missing entirely, not wrong
    for one type)."""
    target = _target()
    script = render_sbatch_script(
        target,
        "det_01",
        "/scratch/exp/runs/det_01",
        "detail.pipeline.detection,detail.pipeline.tracking",
    )
    export_line = "export DETAIL_RUNNER_MODULES=detail.pipeline.detection,detail.pipeline.tracking"
    assert export_line in script
    # the worker must see it before it runs, or the export is dead code
    assert script.index(export_line) < script.index("-m detail.executor.worker")


def test_activation_prefix_matches_the_sbatch_script_environment():
    # A one-off ssh check (the remote model-readiness preflight) must see the same
    # environment a submitted sbatch job runs under, or it answers for the wrong
    # environment.
    target = _target(conda_env="detail-env", modules=["cuda/12.4", "gcc/13.2"])
    prefix = activation_prefix(target)
    script = render_sbatch_script(target, "det_01", "/scratch/exp/runs/det_01", "")
    assert (
        prefix
        == "module load cuda/12.4 && module load gcc/13.2 && conda activate detail-env && "
    )
    # Same lines, same order, as what the sbatch script itself loads/activates —
    # only chained with && for a single ssh command instead of one per script line.
    for line in (
        "module load cuda/12.4",
        "module load gcc/13.2",
        "conda activate detail-env",
    ):
        assert line in script
        assert line in prefix


def test_activation_prefix_empty_when_target_needs_neither():
    target = _target()  # no modules, no conda_env
    assert activation_prefix(target) == ""


def test_submit_script_writes_file_then_sbatches_and_parses_job_id(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ssh_transport,
        "write_remote_file",
        lambda host, path, content, **k: calls.append(("write", host, path)),
    )
    monkeypatch.setattr(
        ssh_transport,
        "run_remote",
        lambda host, command, **k: calls.append(("sbatch", host, command))
        or _FakeResult(stdout="Submitted batch job 123456\n"),
    )
    target = _target()
    job_id = SlurmScheduler().submit_script(
        target, "#!/bin/bash\n...", "/scratch/exp/runs/det_01"
    )
    assert job_id == "123456"
    assert calls[0][0] == "write"
    assert calls[0][2] == "/scratch/exp/runs/det_01/job.sbatch"
    assert "sbatch" in calls[1][2]


def test_submit_script_raises_when_job_id_unparseable(monkeypatch):
    monkeypatch.setattr(ssh_transport, "write_remote_file", lambda *a, **k: None)
    monkeypatch.setattr(
        ssh_transport, "run_remote", lambda *a, **k: _FakeResult(stdout="nonsense\n")
    )
    with pytest.raises(ValueError, match="could not parse"):
        SlurmScheduler().submit_script(_target(), "script", "/scratch/exp/runs/det_01")


@pytest.mark.parametrize(
    "squeue_state,expected",
    [
        ("PENDING", JobState.PENDING),
        ("RUNNING", JobState.RUNNING),
        ("COMPLETING", JobState.RUNNING),
    ],
)
def test_poll_state_reads_squeue_while_queued(monkeypatch, squeue_state, expected):
    monkeypatch.setattr(
        ssh_transport,
        "run_remote",
        lambda host, command, **k: _FakeResult(
            returncode=0, stdout=f"{squeue_state}\n"
        ),
    )
    assert SlurmScheduler().poll_state(_target(), "123456") == expected


@pytest.mark.parametrize(
    "sacct_line,expected",
    [
        ("COMPLETED\n", JobState.COMPLETED),
        ("FAILED\n", JobState.FAILED),
        ("CANCELLED by 12345\n", JobState.CANCELLED),
        ("TIMEOUT\n", JobState.FAILED),
        ("OUT_OF_MEMORY\n", JobState.FAILED),
    ],
)
def test_poll_state_falls_back_to_sacct_once_out_of_squeue(
    monkeypatch, sacct_line, expected
):
    def fake_run(host, command, **k):
        if command.startswith("squeue"):
            return _FakeResult(returncode=1, stdout="")  # not found in queue anymore
        return _FakeResult(returncode=0, stdout=sacct_line)

    monkeypatch.setattr(ssh_transport, "run_remote", fake_run)
    assert SlurmScheduler().poll_state(_target(), "123456") == expected


def test_poll_state_unknown_to_both_reports_failed(monkeypatch):
    monkeypatch.setattr(
        ssh_transport,
        "run_remote",
        lambda host, command, **k: _FakeResult(returncode=1, stdout=""),
    )
    assert SlurmScheduler().poll_state(_target(), "999") == JobState.FAILED


# **Not asking is not an answer** (the class is written up in `test_remote_shepherd.py`). These are
# the two ways of coming back empty told apart: ssh exiting 255 never ran the question, where
# squeue exiting 1 is Slurm saying the job is not in the queue. The control above
# (`..._unknown_to_both_reports_failed`) must keep reporting FAILED, or a job that is genuinely
# gone would be waited on forever.
def _timing_out(*_a, **_k):
    import subprocess

    raise subprocess.TimeoutExpired(cmd="ssh", timeout=30)


def test_poll_state_reports_unreachable_when_ssh_never_connected(monkeypatch):
    asked = []

    def fake_run(host, command, **k):
        asked.append(command.split()[0])
        return _FakeResult(
            returncode=255, stderr="Permission denied (keyboard-interactive)"
        )

    monkeypatch.setattr(ssh_transport, "run_remote", fake_run)

    assert SlurmScheduler().poll_state(_target(), "123456") == JobState.UNREACHABLE
    assert asked == [
        "squeue"
    ], "a second question over a dead connection learns nothing"


def test_poll_state_reports_unreachable_when_the_connection_drops_before_sacct(
    monkeypatch,
):
    def fake_run(host, command, **k):
        if command.startswith("squeue"):
            return _FakeResult(returncode=1, stdout="")  # out of the queue
        return _FakeResult(returncode=255)

    monkeypatch.setattr(ssh_transport, "run_remote", fake_run)

    assert SlurmScheduler().poll_state(_target(), "123456") == JobState.UNREACHABLE


@pytest.mark.parametrize("hangs", ["squeue", "sacct"])
def test_poll_state_reports_unreachable_when_an_ask_hangs(monkeypatch, hangs):
    """A half-dead connection does not fail, it just never answers; unbounded, it stalls the
    shepherd's poll loop for good."""

    def fake_run(host, command, **k):
        assert (
            k.get("timeout_s") == ssh_transport.ASK_TIMEOUT_S
        ), "an ask with no deadline"
        if command.startswith(hangs):
            return _timing_out()
        return _FakeResult(returncode=1, stdout="")

    monkeypatch.setattr(ssh_transport, "run_remote", fake_run)

    assert SlurmScheduler().poll_state(_target(), "123456") == JobState.UNREACHABLE


def test_unreachable_is_never_a_terminal_state():
    from detail.executor.schedulers.base import TERMINAL_STATES

    assert JobState.UNREACHABLE not in TERMINAL_STATES


@pytest.mark.parametrize(
    "result, delivered",
    [
        (_FakeResult(returncode=0), True),
        (
            _FakeResult(returncode=1),
            True,
        ),  # scancel itself refused: already finished, say
        (_FakeResult(returncode=255), False),  # ssh never connected: nothing was sent
    ],
)
def test_cancel_says_whether_the_request_was_delivered(monkeypatch, result, delivered):
    monkeypatch.setattr(ssh_transport, "run_remote", lambda *a, **k: result)

    assert SlurmScheduler().cancel(_target(), "123456") is delivered


def test_cancel_with_an_ask_that_hangs_is_not_delivered(monkeypatch):
    monkeypatch.setattr(ssh_transport, "run_remote", _timing_out)

    assert SlurmScheduler().cancel(_target(), "123456") is False


def test_cancel_issues_scancel_without_raising_on_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ssh_transport,
        "run_remote",
        lambda host, command, **k: calls.append((host, command, k))
        or _FakeResult(returncode=1),
    )
    SlurmScheduler().cancel(_target(), "123456")
    assert calls[0][0] == "tillicum"
    assert "scancel" in calls[0][1] and "123456" in calls[0][1]
    assert calls[0][2].get("check") is False


# --- every sacct state is mapped, and a live one is never a verdict ------------------------- #
@pytest.mark.parametrize("state", slurm.SACCT_STATES)
def test_every_state_sacct_documents_is_mapped(state):
    assert state in slurm._SACCT_STATE_MAP


@pytest.mark.parametrize(
    "sacct_line,expected",
    [
        ("RUNNING\n", JobState.RUNNING),
        ("PENDING\n", JobState.PENDING),
        ("REQUEUED\n", JobState.PENDING),
        ("SUSPENDED\n", JobState.RUNNING),
        ("COMPLETING\n", JobState.RUNNING),
    ],
)
def test_a_running_job_squeue_could_not_report_is_still_running(
    monkeypatch, sacct_line, expected
):
    """The incident: the controller timed out on squeue, sacct said RUNNING, and the job was
    recorded failed while it ran on to complete."""

    def fake_run(host, command, **k):
        if command.startswith("squeue"):
            return _FakeResult(
                returncode=1,
                stderr="slurm_load_jobs error: Socket timed out on send/recv operation",
            )
        return _FakeResult(returncode=0, stdout=f"{sacct_line}RUNNING\nRUNNING\n")

    monkeypatch.setattr(ssh_transport, "run_remote", fake_run)
    state = SlurmScheduler().poll_state(_target(), "351119")
    assert state == expected
    assert state not in TERMINAL_STATES
