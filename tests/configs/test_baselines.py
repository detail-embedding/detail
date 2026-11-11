"""A shipped baseline is data we authored, so a wrong one is a build error.

Baselines (`configs/baselines/*.yaml`) are the named starting points every launch form offers
beside *Defaults*. They are the half of settings reuse that travels: a baseline holds **settings
only**, so it carries no run id and means the same thing in another experiment on another
machine. That is not a nicety — it is the single thing that kills preset systems elsewhere.
ilastik's `.ilp`, TrackMate's `ImagePlus`, a SLEAP config that cannot find its `models/` folder:
each embeds a reference to data, and each becomes unopenable the moment the data moves.

So the rule is enforced at **load** rather than at launch. A baseline naming a selection would
otherwise sit in a dropdown until somebody picked it, and then bind to whatever happened to
share that id — a plausible run, silently the wrong one.

**None ship today**, deliberately: a baseline is a recommendation to every user, and the numbers
for one are measured rather than guessed. That is why the rules below are checked against
fixtures rather than against the directory's contents. A guard that only ever ran over shipped
files would go quiet the moment there were none — testing nothing, in green — which is exactly
the state this package is in. The directory sweep is kept beside them for the day a file lands.
"""

import pytest
import yaml

from detail.configs.baselines import (
    BASELINES_DIR,
    BASELINES_PACKAGE,
    BaselineError,
    _check,
    baselines_for,
    load_baselines,
)


def _shipped_files():
    from importlib.resources import files

    directory = files(BASELINES_PACKAGE) / BASELINES_DIR
    if not directory.is_dir():
        return []
    return [e for e in directory.iterdir() if e.name.endswith((".yaml", ".yml"))]


@pytest.fixture
def baselines_dir(tmp_path, monkeypatch):
    """Point the loader at a directory this test owns.

    Patched at the point of use (`baselines.files`), so the loader under test is the real one —
    the same reader, the same validation, a different directory.
    """
    from detail.configs import baselines as module

    directory = tmp_path / "baselines"
    directory.mkdir()
    monkeypatch.setattr(module, "files", lambda _package: tmp_path)
    monkeypatch.setattr(module, "BASELINES_DIR", "baselines")
    return directory


# --- the rules, on fixtures -------------------------------------------------- #
def test_a_valid_baseline_loads_from_a_file(baselines_dir):
    """The whole path: YAML on disk -> validated -> offered to the modules it names."""
    (baselines_dir / "rig.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "Test rig",
                "settings": {"tracking": {"tracker_id": "iou_single_tracker"}},
            }
        ),
        encoding="utf-8",
    )
    (loaded,) = load_baselines()
    assert loaded.name == "Test rig"
    assert loaded.for_module("tracking") == {"tracker_id": "iou_single_tracker"}
    assert loaded.for_module("detection") is None
    assert [b.name for b in baselines_for("tracking")] == ["Test rig"]
    assert baselines_for("detection") == []


def test_a_baseline_naming_a_selection_is_refused_at_load(baselines_dir):
    """The rule the whole design rests on, checked through the loader rather than `_check`.

    `_check` being right is not enough — it has to be *wired in*, or a bad file still reaches a
    dropdown. `embed_run_id` is an id minted inside one experiment; a baseline carrying one
    would resolve somewhere else to a different run, or to nothing, with no error either way.
    """
    (baselines_dir / "bad.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "Bad",
                "settings": {"temporal_clustering": {"embed_run_id": "emb_02"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BaselineError, match="selection"):
        load_baselines()


def test_an_empty_directory_loads_to_nothing(baselines_dir):
    """The state this package ships in. The picker then offers Defaults and previous runs."""
    assert load_baselines() == []
    assert baselines_for("tracking") == []


def test_a_baseline_naming_an_unknown_parameter_is_refused():
    with pytest.raises(BaselineError, match="not a parameter"):
        _check("X", "tracking", {"no_such_knob": 1})


def test_a_baseline_outside_the_models_bounds_is_refused():
    """A value the model would reject is caught here rather than at validate() time, which is
    after the user has picked it and pressed Run."""
    with pytest.raises(BaselineError):
        _check("X", "detection", {"conf_threshold": 99.0})


def test_a_baseline_for_an_unknown_module_is_refused():
    with pytest.raises(BaselineError, match="not a module"):
        _check("X", "no_such_module", {"anything": 1})


# --- and the directory itself, for the day a file lands ---------------------- #
@pytest.mark.parametrize("entry", _shipped_files(), ids=lambda e: e.name)
def test_a_shipped_baseline_names_only_settings(entry):
    """Every key exists on its module's model, is a `setting`, and is inside its bounds."""
    data = yaml.safe_load(entry.read_text(encoding="utf-8-sig")) or {}
    assert (
        data.get("schema_version") == 1
    ), "schema_version is the first field on every YAML"
    for module_id, values in (data.get("settings") or {}).items():
        _check(data.get("name", entry.name), module_id, values)


def test_whatever_ships_loads():
    """Zero files is the current answer and a fine one; a file that cannot load is not.

    Stated separately from the sweep above because a parametrize over an empty list *passes*,
    silently, and would keep passing if the loader broke.
    """
    assert load_baselines() is not None
    assert len(load_baselines()) == len(_shipped_files())
