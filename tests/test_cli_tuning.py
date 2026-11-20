"""The CLI's surface for tuning detection and tracking settings (§12): `distill run --tune`,
`distill status` on a tuned run, `model tune` and `model tuning`. The computation is stubbed at
``Experiment.tune_tracking``; what is under test is what the commands pass and print."""

import json

from click.testing import CliRunner

from detail.api.experiment import Experiment
from detail.cli import main
from detail.storage import distill_runs as distill_store
from detail.storage import models as models_store
from detail.storage import tunings as tunings_store

SETTING = {
    "detection_conf": 0.05,
    "conf_threshold": 0.1,
    "secondary_conf": 0.05,
    "max_secondary": 4,
}


def _record(tuning_id="tune_001", source=None, created="2026-11-19T10:00:00Z"):
    q = {
        "n_frames": 100,
        "tracked_pct": 100.0,
        "merged_pct": 50.0,
        "grew_pct": 20.0,
        "jumpy_pct": 0.4,
        "median_area": 900,
        "low_conf_pct": 1.0,
        "unassigned_per_frame": 0.1,
    }
    return {
        "id": tuning_id,
        "model_id": "model_001",
        "status": "complete",
        "created_at": created,
        "finished_at": created,
        "source": source or {"kind": "standalone"},
        "recordings": [
            {
                "recording_id": "recording_001",
                "name": "a",
                "frame_rate": 10.0,
                "sampled_frames": 100,
                "windows": [[0, 0, 20]] * 5,
                "multi_subject_share": 0.0,
            }
        ],
        "table": [
            {
                **SETTING,
                "is_default": False,
                "mean_jumpy_pct": 0.4,
                "per_recording": {"recording_001": q},
            },
            {
                "detection_conf": 0.5,
                "conf_threshold": 0.5,
                "secondary_conf": 0.5,
                "max_secondary": 2,
                "is_default": True,
                "mean_jumpy_pct": 3.0,
                "per_recording": {"recording_001": {**q, "jumpy_pct": 3.0}},
            },
        ],
        "pick": {"setting": SETTING, "keep_defaults": False, "reason": "steadiest"},
        "warnings": ["a note"],
    }


def _experiment(tmp_path):
    experiment = Experiment.create("p", tmp_path / "exp")
    directory = models_store.model_dir(experiment.dir, "model_001")
    directory.mkdir(parents=True)
    models_store.write_model_yaml(
        directory,
        {"id": "model_001", "name": "student", "status": "complete", "task": "segment"},
    )
    return experiment


def _write_tuning(experiment, record):
    directory = tunings_store.tuning_dir(experiment.dir, "model_001", record["id"])
    tunings_store.write_tuning(directory, record)
    (directory / "job.json").write_text(
        json.dumps({"schema_version": 1, "status": record["status"]}), encoding="utf-8"
    )


class _Run:
    id = "tune_001"
    status = "complete"
    error = None
    owns_worker = True

    def progress(self):
        return iter(())

    def wait(self):
        return self

    def cancel(self):
        pass

    def record(self):
        return _record()


def test_model_tune_passes_its_options_and_prints_the_flags(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    seen = {}

    def _tune(self, model_id, ids, **kw):
        seen.update(model_id=model_id, ids=ids, **kw)
        return _Run()

    monkeypatch.setattr(Experiment, "tune_tracking", _tune)
    result = CliRunner().invoke(
        main,
        [
            "model",
            "tune",
            "student",
            "--experiment",
            str(experiment.dir),
            "--sample-seconds",
            "60",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["model_id"] == "model_001"  # by name
    assert seen["ids"] is None
    assert seen["sample_seconds"] == 60.0
    assert "recommended: steadiest" in result.output  # the record's own reason
    assert "detail process detection … --conf 0.05" in result.output
    assert (
        "--conf-threshold 0.1 --secondary-conf 0.05 --max-secondary 4" in result.output
    )
    assert "(defaults)" in result.output
    assert "note: a note" in result.output


def test_model_tune_refuses_a_sample_out_of_bounds(tmp_path):
    experiment = _experiment(tmp_path)
    result = CliRunner().invoke(
        main,
        [
            "model",
            "tune",
            "model_001",
            "--experiment",
            str(experiment.dir),
            "--sample-seconds",
            "1",
        ],
    )
    assert result.exit_code != 0
    assert "10.0<=x<=3600.0" in result.output


def test_model_tuning_shows_the_newest_and_says_which(tmp_path):
    experiment = _experiment(tmp_path)
    _write_tuning(
        experiment,
        _record(
            "tune_001",
            {"kind": "distill", "distill_id": "distill_001"},
            "2026-11-18T10:00:00Z",
        ),
    )
    _write_tuning(experiment, _record("tune_002"))
    result = CliRunner().invoke(
        main, ["model", "tuning", "model_001", "--experiment", str(experiment.dir)]
    )
    assert result.exit_code == 0, result.output
    assert "tune_001: complete (from distill distill_001" in result.output
    assert "showing tune_002 (from a standalone tune" in result.output
    assert "the newest of 2" in result.output


def test_model_tuning_of_an_untuned_model_says_how(tmp_path):
    experiment = _experiment(tmp_path)
    result = CliRunner().invoke(
        main, ["model", "tuning", "model_001", "--experiment", str(experiment.dir)]
    )
    assert result.exit_code != 0
    assert "detail model tune model_001" in result.output


def test_distill_run_passes_tune(tmp_path, monkeypatch):
    experiment = _experiment(tmp_path)
    seen = {}

    def _distill(self, **kw):
        seen.update(kw)
        raise ValueError("stop here")

    monkeypatch.setattr(Experiment, "distill", _distill)
    args = ["distill", "run", "--experiment", str(experiment.dir), "--name", "m"]
    args += ["--class", "subject:a subject", "--n-frames", "10"]
    CliRunner().invoke(main, args)
    assert seen["tune"] is False
    CliRunner().invoke(main, args + ["--tune", "--tune-seconds", "90"])
    assert seen["tune"] is True and seen["tune_sample_seconds"] == 90.0


def test_distill_status_prints_a_complete_tune(tmp_path):
    experiment = _experiment(tmp_path)
    _write_tuning(
        experiment, _record(source={"kind": "distill", "distill_id": "distill_001"})
    )
    distill_store.write_run(
        experiment.dir,
        "distill_001",
        {
            "id": "distill_001",
            "name": "m",
            "status": "complete",
            "config": {"tune": {"sample_seconds": 300.0}},
            "stages": {
                "label": {"status": "complete", "dataset_id": "dataset_001"},
                "train": {"status": "complete", "model_id": "model_001"},
                "tune": {
                    "status": "complete",
                    "model_id": "model_001",
                    "tuning_id": "tune_001",
                },
            },
        },
    )
    result = CliRunner().invoke(
        main, ["distill", "status", "distill_001", "--experiment", str(experiment.dir)]
    )
    assert result.exit_code == 0, result.output
    assert "tune: status=complete tuning=tune_001" in result.output
    assert "tuning tune_001 of model_001 (from distill distill_001" in result.output
