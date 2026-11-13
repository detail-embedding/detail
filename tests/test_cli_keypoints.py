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


def test_examples_reject_and_remove_frame(tracked):
    exp = tracked.exp
    _ok("examples", "create", "rats", "--skeleton", "quadruped_side_5", *exp)
    sample = ["--recording", "recording_002", "-n", "3", "--strategy", "spread"]
    _ok("examples", "sample", "ex_01", *sample, *exp)

    out = _ok("examples", "reject", "ex_01", "0", "--bucket", "toward", *exp)
    assert "ex_01: 2 frames, 1 rejected" in out
    assert "1 rejected (never offered again)" in _ok("examples", "show", "ex_01", *exp)
    assert "ex_01: 1 frames" in _ok("examples", "remove-frame", "ex_01", "0", *exp)
    assert "Invalid value for '--bucket'" in _fails(
        "examples", "reject", "ex_01", "0", "--bucket", "sideways", *exp
    )
    assert "no frame 5" in _fails("examples", "remove-frame", "ex_01", "5", *exp)


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
    assert "no metadata column 'box'" in _fails(
        "examples",
        "sample",
        "ex_01",
        "--recording",
        "recording_002",
        "--spread-by",
        "box",
        *exp,
    )


def test_examples_sample_offers_every_strategy_in_the_table():
    from detail.cli_keypoints import examples_sample
    from detail.pipeline import subject_frames

    option = next(p for p in examples_sample.params if p.name == "strategy")
    assert tuple(option.type.choices) == subject_frames.STRATEGIES
    assert option.default == subject_frames.DEFAULT_STRATEGY
    for strategy in subject_frames.STRATEGY_TABLE:
        assert strategy.help in option.help


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
    # the student sees the whole frame unless asked (§12)
    assert (seen["student_input"], seen["student_crop_px"]) == ("whole_frame", 448)
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
            "--student-input",
            "crop",
            "--student-crop-px",
            "320",
        ],
    )
    assert "stop here" in result.output, result.output
    assert (seen["student_input"], seen["student_crop_px"]) == ("crop", 320)


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
    assert "--student-input" in out and "--student-crop-px" in out
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


# --- setup (§26, #717: the remote-detail keypoint setup, on main's classes) -------------- #
def test_setup_start_and_continue_hand_their_flags_to_the_api(tracked, monkeypatch):
    from detail.api.experiment import Experiment

    seen = {}

    def start(self, name, detector_model_id, class_name=None, **options):
        seen.update(name=name, detector=detector_model_id, cls=class_name, **options)
        raise ValueError("stop here")

    def resume(
        self, run_id, example_set=None, score_cutoff=None, class_example_sets=None
    ):
        seen.update(run_id=run_id, example_set=example_set, score_cutoff=score_cutoff)
        raise ValueError("stop here too")

    monkeypatch.setattr(Experiment, "start_keypoint_setup", start)
    monkeypatch.setattr(Experiment, "continue_keypoint_setup", resume)
    result = CliRunner().invoke(
        main,
        [
            "keypoints",
            "setup",
            "start",
            *tracked.exp,
            "--name",
            "rat pose",
            "--detector",
            "model_001",
            "--class",
            "mouse",
            "--teacher",
            "box 1=box1_day2",
            "--window-s",
            "240",
            "--example-set",
            "ex_01",
            "--score-cutoff",
            "0.4",
            "--decode",
            "per_point",
            "--detect-backend",
            "cpu",
            "--student-input",
            "crop",
            "--epochs",
            "3",
        ],
    )
    assert "stop here" in result.output, result.output
    assert (seen["name"], seen["detector"], seen["cls"]) == (
        "rat pose",
        "model_001",
        "mouse",
    )
    assert seen["teachers"] == {"box 1": "box1_day2"} and seen["window_s"] == 240.0
    assert seen["example_set"] == "ex_01" and seen["score_cutoff"] == 0.4
    assert seen["keypoint_settings"] == {"decode": "per_point"}
    assert seen["detect_compute"].backend == "cpu" and seen["train_compute"] is None
    assert seen["student_input"] == "crop" and seen["hyperparams"] == {"epochs": 3}
    assert "--teacher expects SETUP=RECORDING" in _fails(
        "setup",
        "start",
        *tracked.exp,
        "--name",
        "x",
        "--detector",
        "m",
        "--class",
        "c",
        "--teacher",
        "nonsense",
    )
    result = CliRunner().invoke(
        main,
        [
            "keypoints",
            "setup",
            "continue",
            *tracked.exp,
            "--run",
            "distill_001",
            "--example-set",
            "ex_02",
            "--score-cutoff",
            "0.3",
        ],
    )
    assert "stop here too" in result.output, result.output
    assert (seen["run_id"], seen["example_set"], seen["score_cutoff"]) == (
        "distill_001",
        "ex_02",
        0.3,
    )


def test_a_list_method_param_is_a_comma_list():
    """`--method-param track_points=a,b` becomes a list from the field's declared type, not
    from the command knowing the field; `none` is unset (every point)."""
    from detail.cli_keypoints import _method_params

    assert _method_params("temporal_decode", ("track_points=nose, tail_base",)) == {
        "track_points": ["nose", "tail_base"]
    }
    assert _method_params("despike_smooth", ("track_points=none",)) == {
        "track_points": None
    }
    assert _method_params(
        "temporal_decode", ("track_points=snout", "transition_sigma_body=3")
    ) == {"track_points": ["snout"], "transition_sigma_body": 3.0}


def _tune_result():
    from detail.pipeline import keypoint_track_tune as tune

    def setting(name, overrides, pick, check):
        s = tune.TuneSetting(name, overrides)
        s.scores = {
            half: {
                f"group:{g}": {"pck10": {"value": v}}
                for g in ("all", "paired", "unpaired")
            }
            for half, v in (("pick", pick), ("check", check))
        }
        return s

    best = "transition 3, smooth off, despike off"
    return tune.TuneResult(
        detect_runs={"recording_002": "kd_01"},
        example_set_id="ex_02",
        method="temporal_decode",
        base_config={"despike": True, "smooth": True, "missing_penalty": 2.0},
        pick_recordings=["recording_002"],
        check_recordings=["recording_003"],
        coverage={"labelled": 20, "excluded": 1, "uncovered": 0, "scored": 19},
        settings=[
            setting(tune.DETECTION_ONLY, None, 0.74, 0.75),
            setting(tune.CURRENT, {}, 0.59, 0.66),
            setting(
                best,
                {"transition_sigma_body": 3.0, "smooth": False, "despike": False},
                0.79,
                0.80,
            ),
        ],
        best=best,
    )


def test_track_tune_passes_the_detect_runs_and_settings_and_prints_the_recommendation(
    tracked, monkeypatch
):
    from detail.api.experiment import Experiment

    exp = tracked.exp
    assert "exactly one of --detect" in _fails(
        "track-tune", *exp, "--example-set", "ex_02"
    )
    seen = {}

    def capture(self, example_set_id, **options):
        seen.update(example_set_id=example_set_id, **options)
        return _tune_result()

    monkeypatch.setattr(Experiment, "tune_keypoint_tracking", capture)
    out = _ok(
        "track-tune",
        *exp,
        "--example-set",
        "ex_02",
        "--detect",
        "kd_01",
        "--method-param",
        "missing_penalty=5",
    )
    assert seen["example_set_id"] == "ex_02" and seen["detect_run_id"] == "kd_01"
    assert seen["base_config"] == {"missing_penalty": 5.0} and seen["seed"] == 0
    assert "excluded (the detect run learned from them) 1" in out
    assert (
        "* transition 3, smooth off, despike off" in out
    )  # the recommendation, marked
    assert "80.0%" in out  # its check-half figure
    assert (
        "detail keypoints track --detect kd_01 --method temporal_decode "
        "--method-param despike=false --method-param smooth=false "
        "--method-param transition_sigma_body=3.0" in out
    )  # only what differs from the method's defaults


def _preview_result():
    from detail.core.skeleton import Skeleton
    from detail.pipeline import keypoint_track_preview as preview

    figures = {
        "placed": 1.0,
        "median_move": 0.132,
        "jumps_per_min": 3268.9,
        "swaps_per_min": 222.74,
        "stretch": 0.357,
        "agreement": 0.948,
    }

    def setting(name, overrides, move):
        s = preview.PreviewSetting(name, overrides)
        s.figures = {preview.ALL: {**figures, "median_move": move}}
        return s

    return preview.PreviewResult(
        method="temporal_decode",
        base_config={"despike": True, "smooth": True, "transition_sigma_body": 0.15},
        skeleton=Skeleton(names=("nose", "tail"), edges=(("nose", "tail"),)),
        tracked=("nose", "tail"),
        windows=[
            preview.PreviewWindow(
                "recording_002", "part_001", 0, 3 * 10**9, 5 * 10**9, "kd_01",
                "track_01", 30.0, (40, 40),
            )  # fmt: skip
        ],
        settings=[
            setting(preview.DETECTION_ONLY, None, 0.132),
            setting(preview.CURRENT, {}, 0.004),
            setting("transition 3, smooth off", {"transition_sigma_body": 3.0, "smooth": False}, 0.011),  # fmt: skip
        ],
    )


def test_track_preview_passes_the_windows_and_variants_and_prints_the_table(
    tracked, monkeypatch
):
    from detail.api.experiment import Experiment

    exp = tracked.exp
    assert "exactly one of --detect" in _fails("track-preview", *exp)
    assert "need --method" in _fails(
        "track-preview", *exp, "--detect", "kd_01", "--variant", "smooth=false"
    )
    seen = {}

    def capture(self, config, recordings=None, **options):
        seen.update(config=config, recordings=recordings, **options)
        return _preview_result()

    monkeypatch.setattr(Experiment, "preview_keypoint_tracking", capture)
    out = _ok(
        "track-preview",
        *exp,
        "--detect",
        "kd_01",
        "--recording",
        "recording_002",
        "--start",
        "recording_002=3",
        "--duration",
        "2",
        "--method",
        "temporal_decode",
        "--variant",
        "transition_sigma_body=3 smooth=false",
    )
    assert seen["config"]["detect_run_id"] == "kd_01"
    assert seen["config"]["method"] == "temporal_decode"
    assert seen["recordings"] == ["recording_002"]
    assert seen["starts"] == {"recording_002": 3.0} and seen["duration_s"] == 2.0
    assert seen["variants"] == [{"transition_sigma_body": 3.0, "smooth": False}]
    assert "3.0-5.0 s (2.0 s)" in out
    assert "they do not\nrank" in out or "they do not rank" in out
    lines = out.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("setting"))
    assert lines[first + 1].startswith("detection only")  # the floor, first
    assert "13.2%" in lines[first + 1] and "3268.90" in lines[first + 1]
    assert (
        "track with transition 3, smooth off: detail keypoints track --detect kd_01 "
        "--method temporal_decode --method-param smooth=false "
        "--method-param transition_sigma_body=3.0 --recording ..." in out
    )
    assert "bad --start" in _fails(
        "track-preview", *exp, "--detect", "kd_01", "--start", "recording_002"
    )


def test_track_preview_grid_marks_the_front_and_offers_its_commands(
    tracked, monkeypatch
):
    import dataclasses

    from detail.api.experiment import Experiment
    from detail.pipeline import keypoint_track_preview as preview

    exp = tracked.exp
    assert "--grid or --variant, not both" in _fails(
        "track-preview", *exp, "--detect", "kd_01", "--method", "temporal_decode",
        "--variant", "smooth=false", "--grid",
    )  # fmt: skip
    base = _preview_result()
    beaten = preview.PreviewSetting("transition 0.15, smooth off, despike off", {"transition_sigma_body": 0.15, "smooth": False, "despike": False})  # fmt: skip
    beaten.figures = {preview.ALL: {**base.settings[2].figures[preview.ALL], "agreement": 0.5}}  # fmt: skip
    result = dataclasses.replace(
        base,
        settings=base.settings[:2] + [beaten, base.settings[2]],
        grid=True,
        front=["transition 3, smooth off"],
    )
    seen = {}

    def capture(self, config, recordings=None, **options):
        seen.update(options)
        return result

    monkeypatch.setattr(Experiment, "preview_keypoint_tracking", capture)
    out = _ok("track-preview", *exp, "--detect", "kd_01", "--grid")
    assert seen["grid"] is True and seen["variants"] == []
    assert "on the front (1 of 3)" in out and "not yet checked against labels" in out
    lines = out.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("setting"))
    # detection, current, then the front (marked), then the rest
    assert lines[first + 1].lstrip().startswith("detection only")
    assert lines[first + 2].lstrip().startswith("current settings")
    assert lines[first + 3].startswith("* transition 3, smooth off")
    assert lines[first + 4].lstrip().startswith("transition 0.15")
    assert "94.8%" in lines[first + 3]
    # a command for the current settings and the front, none for what the front beats
    assert "track with transition 3, smooth off:" in out
    assert "track with current settings:" in out
    assert "track with transition 0.15" not in out
