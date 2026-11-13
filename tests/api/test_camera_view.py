"""The camera view through the API (§10): an experiment default every recording inherits, and a
recording override that clears back to inheriting.

Persisted the way every YAML is: ``schema_version: 1`` stays the literal first key after a
write, and a refused value leaves the file as it was.
"""

import pytest
import yaml

from detail.api.experiment import Experiment


@pytest.fixture
def experiment(tmp_path):
    return Experiment.create("cv", tmp_path)


def test_recordings_inherit_the_experiment_default(experiment):
    rec = experiment.create_recording("r1")
    assert not rec.camera_view.is_set
    experiment.set_camera_view("side")
    assert rec.camera_view.view == "side"
    assert rec.camera_view.source == "experiment"
    assert rec.default_camera_view.view == "side"


def test_an_override_wins_and_clearing_it_inherits_again(experiment):
    experiment.set_camera_view("side")
    rec = experiment.create_recording("r1")
    rec.set_camera_view("other", "45° from above")
    assert (rec.camera_view.view, rec.camera_view.source) == ("other", "recording")
    rec.set_camera_view(None)
    assert (rec.camera_view.view, rec.camera_view.source) == ("side", "experiment")
    data = yaml.safe_load((rec.dir / "recording.yaml").read_text(encoding="utf-8-sig"))
    assert "camera_view" not in data and "camera_view_note" not in data


def test_changing_the_default_moves_only_inheriting_recordings(experiment):
    experiment.set_camera_view("side")
    inherits = experiment.create_recording("a")
    own = experiment.create_recording("b")
    own.set_camera_view("top_down")
    experiment.set_camera_view("angled_down")
    assert inherits.camera_view.view == "angled_down"
    assert own.camera_view.view == "top_down"


def test_schema_version_stays_first_after_a_write(experiment):
    experiment.set_camera_view("side")
    rec = experiment.create_recording("r1")
    rec.set_camera_view("top_down")
    for path in (experiment.dir / "config.yaml", rec.dir / "recording.yaml"):
        first = path.read_text(encoding="utf-8-sig").splitlines()[0]
        assert first == "schema_version: 1"


def test_a_refused_value_leaves_the_file_unchanged(experiment):
    experiment.set_camera_view("side")
    before = (experiment.dir / "config.yaml").read_text(encoding="utf-8-sig")
    with pytest.raises(ValueError, match="needs a description"):
        experiment.set_camera_view("other")
    assert (experiment.dir / "config.yaml").read_text(encoding="utf-8-sig") == before


def test_clearing_the_default_leaves_recordings_not_set(experiment):
    experiment.set_camera_view("side")
    rec = experiment.create_recording("r1")
    experiment.set_camera_view(None)
    assert not experiment.camera_view.is_set
    assert not rec.camera_view.is_set
