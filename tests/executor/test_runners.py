from pathlib import Path

from detail.executor.runners import JOB_RUNNERS, RunnerSpec, register_runner


def test_register_runner_stores_runner_and_resolver():
    def resolver(experiment_root, job):
        return Path(experiment_root)

    @register_runner("ttype", run_dir_for=resolver)
    def run(job, run_dir, progress_cb, stop_event):
        pass

    try:
        spec = JOB_RUNNERS["ttype"]
        assert isinstance(spec, RunnerSpec)
        assert spec.run is run
        assert spec.run_dir_for is resolver
    finally:
        JOB_RUNNERS.pop("ttype", None)
