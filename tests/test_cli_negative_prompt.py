"""The CLI reaches the negative prompt and shows it back (§12 *Negative prompt*).

The two launch commands hand ``--negative-prompt`` to the API by name; the fake experiment
records what arrived and then refuses, so nothing runs. The three read-back commands print
the dataset's or model's negative classes from a real record.
"""

import types

import pytest
from click.testing import CliRunner

from detail import cli
from detail.api import models as api_models
from detail.api.experiment import Experiment
from detail.cli import main
from detail.models import distilled
from detail.storage import datasets as datasets_store
from detail.storage import models as models_store

NEGATIVE_LINE = "cable (learned apart, never reported)"


class _Recorded(ValueError):
    pass


@pytest.fixture
def calls(monkeypatch):
    seen = {}

    def _record(name):
        def method(*args, **kwargs):
            seen[name] = kwargs
            raise _Recorded("stop here")

        return method

    proj = types.SimpleNamespace(
        distill=_record("distill"), label_dataset=_record("label_dataset")
    )
    monkeypatch.setattr(cli, "_open_experiment", lambda _d: proj)
    return seen


def _invoke(*args):
    return CliRunner().invoke(main, list(args))


@pytest.mark.parametrize(
    "command, method",
    [(("distill", "run"), "distill"), (("dataset", "label"), "label_dataset")],
)
@pytest.mark.parametrize(
    "flag, expected", [((), None), (("--negative-prompt", "cable"), "cable")]
)
def test_the_launch_commands_pass_the_negative_prompt(
    calls, tmp_path, command, method, flag, expected
):
    _invoke(
        *command, "--experiment", str(tmp_path), "--name", "m",
        "--class", "mouse:a mouse", "--n-frames", "4", *flag,
    )  # fmt: skip
    assert calls[method]["negative_prompt"] == expected


def _model(proj) -> str:
    mdir = models_store.model_dir(proj.dir, "model_001")
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "best.pt").write_bytes(b"weights")
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "tethered",
            "model_family": "yolo",
            "task": "segment",
            "classes": [{"id": 0, "name": "mouse"}],
            "negative_classes": [{"id": 1, "name": "cable"}],
            "weights": {"best": "run/weights/best.pt"},
            "status": "complete",
        },
    )
    return "model_001"


def test_dataset_status_shows_the_negative_prompt(tmp_path):
    proj = Experiment.create("p", tmp_path)
    ddir = datasets_store.dataset_dir(proj.dir, "dataset_001")
    ddir.mkdir(parents=True)
    datasets_store.write_dataset_yaml(
        ddir,
        {
            "id": "dataset_001",
            "name": "d",
            "status": "complete",
            "classes": [{"id": 0, "name": "mouse"}],
            "negative_classes": [{"id": 1, "name": "cable"}],
        },
    )
    result = _invoke("dataset", "status", "dataset_001", "--experiment", str(proj.dir))
    assert result.exit_code == 0, result.output
    assert "  classes: mouse\n" in result.output
    assert f"  negative prompt: {NEGATIVE_LINE}\n" in result.output


def test_train_status_shows_the_negative_prompt(tmp_path):
    proj = Experiment.create("p", tmp_path)
    model_id = _model(proj)
    result = _invoke("train", "status", model_id, "--experiment", str(proj.dir))
    assert result.exit_code == 0, result.output
    assert f"  negative prompt: {NEGATIVE_LINE}\n" in result.output


def test_model_show_shows_the_negative_prompt(tmp_path):
    proj = Experiment.create("p", tmp_path)
    gid = distilled.register_distilled_model(
        proj.dir, _model(proj), source_experiment_name="p", source_experiment_id="p"
    )
    assert api_models.imported_model(gid).negative_classes == ["cable"]
    result = _invoke("model", "show", gid)
    assert result.exit_code == 0, result.output
    assert "classes:     mouse\n" in result.output
    assert f"negative:    {NEGATIVE_LINE}\n" in result.output
