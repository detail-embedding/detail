"""The ``detail keypoints`` group (§26, PR 2): ``skeleton`` and ``examples`` subcommands.

Each command is driven through the real root group, so its options parse and its errors reach
the user as a clean message rather than a traceback."""

import pytest
import yaml
from click.testing import CliRunner

from detail.cli import main
from detail.core.skeleton import PRESETS
from tests import experiments

N_FRAMES = 5


def _run(*args):
    return CliRunner().invoke(main, ["keypoints", *args])


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
def tracked(processed_recording):
    built = processed_recording
    detections, object_ids = [], []
    for frame in range(N_FRAMES):
        c0 = 4 + frame * 2
        bbox = (c0 / 40, 0.25, (c0 + 8 + frame * 4) / 40, 0.75)
        detections.append(
            (frame, 0, bbox, built.square(10, 30, c0, c0 + 8 + frame * 4))
        )
        object_ids.append(0)
    recording = built.add_recording("recording_002", detections, object_ids)
    experiments.write_video(recording.dir / "video" / "part_001.mp4", N_FRAMES)
    built.exp = ["--experiment", str(built.experiment.dir)]
    return built


# --- create, edit, duplicate, export --------------------------------------------- #
def test_create_edit_duplicate_export_and_import(tmp_path):
    from detail.api.experiment import Experiment

    proj = Experiment.create("p", tmp_path)
    exp = ["--experiment", str(proj.dir)]
    assert "created example set: ex_01" in _ok(
        "examples", "create", "rats", "--skeleton", "quadruped_side_5", *exp
    )
    assert "updated example set: ex_01" in _ok(
        "examples", "edit", "ex_01", "--name", "side rats", *exp
    )
    es = proj.example_set("ex_01")
    assert es.name == "side rats" and es.skeleton == PRESETS["quadruped_side_5"][1]
    assert es.skeleton_ref["id"] == "quadruped_side_5"
    assert "nothing to change" in _fails("examples", "edit", "ex_01", *exp)
    assert "updated example set: ex_01" in _ok(
        "examples", "edit", "ex_01", "--skeleton", "midline_3", *exp
    )
    assert proj.example_set("ex_01").skeleton == PRESETS["midline_3"][1]
    _ok("examples", "edit", "ex_01", "--skeleton", "quadruped_side_5", *exp)
    assert "No such option '--rename'" in _fails(
        "examples", "edit", "ex_01", "--name", "x", "--rename", "a=b", *exp
    )

    assert "created example set: ex_02" in _ok("examples", "duplicate", "ex_01", *exp)
    assert proj.example_set("ex_02").name == "side rats copy"
    assert "created example set: ex_03" in _ok(
        "examples", "create", "copied", "--skeleton", "ex_01", *exp
    )

    out = tmp_path / "shared"
    assert "examples.yaml" in _ok(
        "examples", "export", "ex_01", "--out", str(out), *exp
    )
    assert "choose an empty folder" in _fails(
        "examples", "export", "ex_01", "--out", str(out), *exp
    )
    assert "imported example set: ex_04 (0 frames)" in _ok(
        "examples", "import", str(out / "examples.yaml"), *exp
    )
    sk_file = tmp_path / "two.yaml"
    sk_file.write_text(
        yaml.dump({"names": ["a", "b"], "pairs": [["a", "b"]]}), encoding="utf-8"
    )
    assert "imported example set: ex_05 (0 frames)" in _ok(
        "examples", "import", str(sk_file), *exp
    )
    assert "no skeleton preset, library skeleton, example set or file 'nope'" in _fails(
        "examples", "create", "x", "--skeleton", "nope", *exp
    )


# --- examples --------------------------------------------------------------------- #
def test_examples_create_sample_show_and_delete(tracked):
    exp = tracked.exp
    assert "no example sets" in _ok("examples", "list", *exp)
    assert "created example set: ex_01" in _ok(
        "examples", "create", "rats", "--skeleton", "quadruped_side_5", *exp
    )
    assert "ex_01  rats  example  0 frames, 0 marked  skeleton 5 points" in _ok(
        "examples", "list", *exp
    )

    out = _ok(
        "examples",
        "sample",
        "ex_01",
        "--recording",
        "recording_002",
        "-n",
        "2",
        "--strategy",
        "spread",
        *exp,
    )
    assert "ex_01: 2 frames" in out

    shown = _ok("examples", "show", "ex_01", *exp)
    assert (
        "skeleton (" in shown
        and "nose, left_ear, right_ear, mid_back, tail_base" in shown
    )
    assert "pairs: left_ear/right_ear" in shown
    assert shown.count("pending: 0 placed") == 2
    assert "(spread)" in shown
    assert "2 frames: 0 marked, 2 pending" in shown

    assert "deleted example set: ex_01" in _ok("examples", "delete", "ex_01", *exp)
    assert "no example set 'ex_01'" in _fails("examples", "show", "ex_01", *exp)


def test_examples_add_frame_and_its_refusals(tracked):
    exp = tracked.exp
    _ok("examples", "create", "rats", "--skeleton", "quadruped_side_5", *exp)
    frame = [
        "--recording",
        "recording_002",
        "--tracking-run",
        "track_01",
        "--part",
        "part_001",
        "--track",
        "0",
    ]
    assert "ex_01: 1 frames" in _ok(
        "examples", "add-frame", "ex_01", *frame, "--frame", "3", *exp
    )
    assert "already in example set" in _fails(
        "examples", "add-frame", "ex_01", *frame, "--frame", "3", *exp
    )
    assert "no subject on frame 40" in _fails(
        "examples", "add-frame", "ex_01", *frame, "--frame", "40", *exp
    )
    assert "(picked)" in _ok("examples", "show", "ex_01", *exp)


def test_examples_sample_errors_are_clean(tracked):
    exp = tracked.exp
    _ok("examples", "create", "rats", "--skeleton", "quadruped_side_5", *exp)
    assert "no example set 'ex_09'" in _fails(
        "examples", "sample", "ex_09", "--recording", "recording_002", *exp
    )
    assert "nowhere" in _fails(
        "examples",
        "sample",
        "ex_01",
        "--recording",
        "nowhere",
        "--strategy",
        "spread",
        *exp,
    )
    assert "no skeleton preset, library skeleton, example set or file 'nope'" in _fails(
        "examples", "create", "x", "--skeleton", "nope", *exp
    )


def test_examples_import_a_prepared_yaml(tracked, tmp_path):
    exp = tracked.exp
    proj = tracked.experiment
    _ok("examples", "create", "rats", "--skeleton", "quadruped_side_5", *exp)
    _ok(
        "examples",
        "sample",
        "ex_01",
        "--recording",
        "recording_002",
        "-n",
        "1",
        "--strategy",
        "spread",
        *exp,
    )
    from detail.storage import example_sets as store

    src = store.set_dir(proj.dir, "ex_01")
    body = yaml.safe_load((src / "examples.yaml").read_text(encoding="utf-8-sig"))
    share = tmp_path / "share"
    (share / "frames").mkdir(parents=True)
    image = body["frames"][0]["image"]
    (share / image).write_bytes((src / image).read_bytes())
    (share / "examples.yaml").write_text(
        yaml.dump(body, sort_keys=False), encoding="utf-8"
    )

    assert "imported example set: ex_02 (1 frames)" in _ok(
        "examples", "import", str(share / "examples.yaml"), *exp
    )
    assert len(proj.example_sets()) == 2


def test_help_names_both_groups():
    out = _ok("--help")
    commands = out.split("Commands:")[1]
    assert (
        "examples" in commands and "\n  skeleton " not in commands
    ), "skeletons are machine-wide: `detail skeleton`, not `detail keypoints skeleton`"
    assert "sample" in _ok("examples", "--help")
    assert "presets" not in _ok("examples", "--help"), "the library lists them now"


# --- distill (PR 8) -------------------------------------------------------------- #
def test_distill_and_dataset_keypoints_refuse_cleanly_without_a_detect_run(tracked):
    track = ["--teacher", "detect"]
    out = _fails(
        "distill",
        *tracked.exp,
        *track,
        "--name",
        "s",
        "--recording",
        "recording_002",
        "--n-frames",
        "4",
    )
    assert "no complete keypoint detect run" in out
    result = CliRunner().invoke(
        main,
        [
            "dataset",
            "keypoints",
            *tracked.exp,
            *track,
            "--name",
            "s",
            "--recording",
            "recording_002",
            "--n-frames",
            "4",
        ],
    )
    assert result.exit_code != 0 and "Traceback" not in result.output
    assert "no complete keypoint detect run" in result.output
    assert "name at least one teacher detect run" in _fails(
        "distill", *tracked.exp, *track, "--name", "s", "--n-frames", "4"
    )
    assert "RECORDING:kd_NN" in _fails(
        "distill",
        *tracked.exp,
        *track,
        "--name",
        "s",
        "--detect-run",
        "bad",
        "--n-frames",
        "4",
    )
    assert "Missing option '--teacher'" in _fails(
        "distill", *tracked.exp, "--name", "s", "--n-frames", "4"
    )


def test_the_patch_teacher_options_refuse_a_mix_and_reach_the_api(tracked, monkeypatch):
    from detail.api.experiment import Experiment

    patch = ["--teacher", "patch", "--name", "s", "--n-frames", "4"]
    assert "give --example-set" in _fails(
        "distill", *tracked.exp, *patch, "--recording", "recording_002"
    )
    assert "--detect-run is for --teacher detect" in _fails(
        "distill",
        *tracked.exp,
        *patch,
        "--example-set",
        "ex_01",
        "--detect-run",
        "recording_002:kd_01",
    )
    assert "are for --teacher patch" in _fails(
        "distill",
        *tracked.exp,
        "--teacher",
        "detect",
        "--name",
        "s",
        "--n-frames",
        "4",
        "--example-set",
        "ex_01",
    )
    assert "RECORDING:track_NN" in _fails(
        "distill", *tracked.exp, *patch, "--example-set", "ex_01", "--tracking-run", "x"
    )
    seen = {}

    def capture(self, name, detect_runs=None, n_frames=None, **options):
        seen.update(name=name, detect_runs=detect_runs, n_frames=n_frames, **options)
        raise ValueError("stop here")

    monkeypatch.setattr(Experiment, "keypoint_dataset", capture)
    result = CliRunner().invoke(
        main,
        [
            "dataset",
            "keypoints",
            *tracked.exp,
            *patch,
            "--example-set",
            "ex_01",
            "--tracking-run",
            "recording_002:track_01",
            "--score-cutoff",
            "0.5",
            "--label-backend",
            "cpu",
        ],
    )
    assert result.exit_code != 0 and "stop here" in result.output
    assert seen["example_set"] == "ex_01" and seen["detect_runs"] is None
    assert seen["tracking"] == {"recording_002": "track_01"}
    assert seen["model_id"] == "dinov3_match" and seen["score_cutoff"] == 0.5
    assert seen["label_compute"].backend == "cpu"


def test_the_detect_teacher_names_its_tracking_run_only_when_asked(
    tracked, monkeypatch
):
    """§12 revised 2026-10-21: a detect run's poses are given to a tracking run's subjects.
    ``--tracking-run`` names it for a recording with several; left out, nothing is sent and
    the API resolves each recording's one tracking run on the detect run's detection run.
    """
    from detail.api.experiment import Experiment

    seen = {}

    def capture(self, name, detect_runs=None, n_frames=None, **options):
        seen.clear()
        seen.update(name=name, detect_runs=detect_runs, **options)
        raise ValueError("stop here")

    monkeypatch.setattr(Experiment, "keypoint_dataset", capture)
    base = [
        "dataset",
        "keypoints",
        *tracked.exp,
        "--teacher",
        "detect",
        "--name",
        "s",
        "--n-frames",
        "4",
        "--recording",
        "recording_002",
        "--detect-run",
        "recording_003:kd_02",
    ]
    result = CliRunner().invoke(main, base)
    assert "stop here" in result.output
    assert seen["detect_runs"] == {"recording_002": None, "recording_003": "kd_02"}
    assert "tracking" not in seen
    result = CliRunner().invoke(
        main, base + ["--tracking-run", "recording_003:track_02"]
    )
    assert "stop here" in result.output
    assert seen["tracking"] == {"recording_003": "track_02"}
    assert "RECORDING:track_NN" in _fails(
        "distill", *base[2:], "--tracking-run", "track_02"
    )


def test_help_names_distill_and_dataset_keypoints():
    out = _ok("distill", "--help")
    assert (
        "--score-cutoff" in out and "--base-model" in out and "--train-backend" in out
    )
    result = CliRunner().invoke(main, ["dataset", "keypoints", "--help"])
    assert result.exit_code == 0, result.output
    assert "--detect-run" in result.output and "--base-model" not in result.output
    assert "--teacher" in result.output and "--example-set" in result.output


# --- audit sets an earlier version made ------------------------------------------- #
def test_an_audit_set_left_on_disk_still_lists_and_deletes(tmp_path):
    """`keypoints audit` is gone; the sets it made are not.

    An audit set is an example set with ``purpose: audit``, so it is still in the experiment
    after the group that made it was removed (``docs/removed_capabilities.md``). It has to
    stay out of the default listing — a method must never learn from it — and stay reachable:
    `--purpose all` shows it, labelled, and `examples delete` removes it.
    """
    from detail.api.experiment import Experiment
    from detail.storage import example_sets

    proj = Experiment.create("p", tmp_path)
    exp = ["--experiment", str(proj.dir)]
    audit = example_sets.create_set(
        proj.dir, "rats audit", PRESETS["quadruped_side_5"][1], purpose="audit"
    )

    assert audit.id not in _ok("examples", "list", *exp)
    listed = _ok("examples", "list", "--purpose", "all", *exp)
    assert f"{audit.id}  rats audit  audit" in listed
    _fails("examples", "list", "--purpose", "audit", *exp)  # no longer a choice
    _ok("examples", "delete", audit.id, *exp)
    assert audit.id not in _ok("examples", "list", "--purpose", "all", *exp)
