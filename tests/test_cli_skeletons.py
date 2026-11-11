"""The ``detail skeleton`` group (§26 *The Keypoints manager*), driven through the real root
group so options parse and refusals reach the user as a clean message."""

import pytest
import yaml
from click.testing import CliRunner

from detail.api.experiment import Experiment
from detail.cli import main
from detail.core.skeleton import PRESETS, Skeleton
from detail.storage import locations


def _run(*args):
    return CliRunner().invoke(main, ["skeleton", *args])


def _ok(*args) -> str:
    result = _run(*args)
    assert result.exit_code == 0, result.output
    return result.output


def _fails(*args) -> str:
    result = _run(*args)
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    return result.output


@pytest.fixture
def root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(locations, "experiments_root", lambda: workspace)
    return workspace


def test_the_group_holds_exactly_these_commands():
    from detail.cli_skeletons import skeleton

    assert set(skeleton.commands) == {
        "list",
        "show",
        "create",
        "edit",
        "save-as-new",
        "duplicate",
        "export",
        "import",
        "delete",
    }
    assert main.commands["skeleton"] is skeleton


def test_list_create_show_edit_and_delete(root, tmp_path):
    out = _ok("list")
    for preset_id, (label, sk) in PRESETS.items():
        assert f"{preset_id}  {label}  {len(sk)} points" in out
    assert "created skeleton: sk_01" in _ok(
        "create", "rats", "--from", "quadruped_side_5"
    )
    shown = _ok("show", "sk_01")
    assert "rats  5 points" in shown and "used by: nothing" in shown
    edited = tmp_path / "edited.yaml"
    edited.write_text(
        yaml.dump(Skeleton(names=("a", "b"), edges=(("a", "b"),)).to_dict()),
        encoding="utf-8",
    )
    assert "updated skeleton: sk_01" in _ok(
        "edit", "sk_01", "--name", "two", "--skeleton", str(edited)
    )
    assert "nothing to change" in _fails("edit", "sk_01")
    assert "built-in preset" in _fails("edit", "quadruped_side_5", "--name", "x")
    assert "deleted skeleton: sk_01" in _ok("delete", "sk_01")
    assert "no skeleton 'sk_01'" in _fails("show", "sk_01")


def test_a_used_skeleton_refuses_edit_and_delete_and_says_what_uses_it(root, tmp_path):
    _ok("create", "rats", "--from", "quadruped_side_5")
    exp = Experiment.create("lab", root)
    exp.create_example_set("overhead", "sk_01")
    assert "used by: example set examples/ex_01/examples.yaml (lab)" in _ok(
        "show", "sk_01"
    )
    edited = tmp_path / "edited.yaml"
    edited.write_text(yaml.dump(Skeleton(names=("a",)).to_dict()), encoding="utf-8")
    assert "save the change as a new skeleton" in _fails(
        "edit", "sk_01", "--skeleton", str(edited)
    )
    assert "delete those first" in _fails("delete", "sk_01")
    assert "created skeleton: sk_02" in _ok(
        "save-as-new", "sk_01", "rats v2", "--skeleton", str(edited)
    )
    assert "updated skeleton: sk_01" in _ok("edit", "sk_01", "--name", "renamed")


def test_duplicate_export_import(root, tmp_path):
    assert "created skeleton: sk_01" in _ok("duplicate", "midline_3", "--name", "worm")
    out = tmp_path / "worm.yaml"
    assert "exported skeleton" in _ok("export", "sk_01", str(out))
    assert "already exists" in _fails("export", "sk_01", str(out))
    assert "imported skeleton: sk_02" in _ok("import", str(out), "--name", "worm 2")
