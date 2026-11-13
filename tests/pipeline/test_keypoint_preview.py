"""Testing a keypoint teacher before a distill (§12, *Test cut-off…*): frames drawn the way the
distill would, the teacher's placements on them, the held-out report in pixels, and nothing
written. The teacher is the fake keypoint method the detect tests use (an external inference
call is the one legitimate stand-in); the recording, zarrs, video and runs are real."""

import numpy as np
import pytest

from detail.pipeline import keypoint_preview as preview
from tests.pipeline.test_keypoint_dataset import (  # noqa: F401 — fixtures
    patch_teacher,
    taught,
)
from tests.pipeline.test_keypoint_detect import (
    tracked,
)  # noqa: F401 — `taught` needs it

DETECT = {
    "kind": "keypoint_detect",
    "detect_runs": {"recording_002": "kd_01"},
    "tracking_runs": {"recording_002": "track_01"},
    "match": {"min_inside": 0.5, "mask_grow": 0.05},
}


def _row(point, score, error_px, body_px=100.0):
    return {
        "frame": 0,
        "point": point,
        "score": score,
        "error_px": error_px,
        "body_px": body_px,
    }


def test_the_report_splits_at_the_cutoff_in_pixels_and_body_share():
    rows = [_row("nose", 0.9, 2.0) for _ in range(8)] + [
        _row("nose", 0.2, 30.0) for _ in range(3)
    ]
    report = preview.cutoff_report(rows, 0.5)
    above = report["overall"]["above"]
    assert above == {"n": 8, "median_px": 2.0, "median_pct": 2.0, "near": 1.0}
    assert report["overall"]["below"] == {
        "n": 3,
        "median_px": None,
        "median_pct": None,
        "near": None,
    }, "fewer than 8 placements: the count, never a median"
    assert set(report["points"]) == {"nose"}
    everything = preview.cutoff_report(rows, None)["overall"]
    assert everything["above"]["n"] == 11 and everything["below"]["n"] == 0
    missed = preview.cutoff_report([_row("tail", float("nan"), float("nan"))], 0.5)
    assert missed["overall"]["below"]["n"] == 0, "a point never placed is not an error"


def test_a_detect_run_teacher_is_tested_on_frames_drawn_like_the_distills(taught):
    it, recording = taught
    before = sorted(p.name for p in it.experiment.dir.rglob("*"))
    stages = []
    result = preview.run_teacher_test(
        it.experiment.dir,
        DETECT,
        n_frames=2,
        progress=lambda stage, fraction, detail: stages.append(stage),
    )
    assert (
        sorted(p.name for p in it.experiment.dir.rglob("*")) == before
    ), "writes nothing"
    frames = sorted(f.frame_idx for f in result.frames)
    assert (
        len(frames) == 2 and frames[0] in (0, 1) and frames[1] in (2, 3)
    ), "one frame from each half of the complete frames, spread over time"
    assert result.method == "fake_kp" and result.held_out is None, "no example set"
    assert result.score_meaning == "heatmap peak"
    first = result.frames[0]
    assert first.image.shape == (40, 40, 3)
    assert all(len(t[1]) == 2 for t in first.tracks.values())
    assert result.taught_visible(0.4) == result.taught_visible(None), "scores are 0.5"
    assert result.taught_visible(0.6)[0] == 0
    assert stages[0] == "reading tracked frames" and stages[-1] == "complete"
    assert "assigning teacher instances" in stages, "the drawn frames are assigned"
    other = preview.run_teacher_test(it.experiment.dir, DETECT, n_frames=2, seed=7)
    assert len(other.frames) == 2


def test_the_patch_teacher_labels_the_test_frames_itself(patch_teacher):
    it, _recording, set_id, seen = patch_teacher
    from detail.api import keypoint_distill

    teacher, compute = keypoint_distill.resolve_teacher(
        it.experiment,
        example_set=set_id,
        tracking=["recording_002"],
        model_id="fake_learner",
    )
    stages = []
    result = preview.run_teacher_test(
        it.experiment.dir,
        teacher,
        compute=compute,
        n_frames=4,
        progress=lambda stage, fraction, detail: stages.append(stage),
    )
    assert sorted(f.frame_idx for f in result.frames) == [0, 1, 2, 3]
    assert seen["devices"] == ["cpu"] and all(seen["masks"])
    assert "loading fake_learner" in stages
    assert result.held_out is None, "the fake has no score_held_out"
    assert np.isfinite(result.scores()).all() and len(result.scores()) > 0


def test_a_test_needs_a_frame(taught):
    it, _recording = taught
    with pytest.raises(ValueError, match="n_frames must be 1"):
        preview.run_teacher_test(it.experiment.dir, DETECT, n_frames=0)


@pytest.mark.parametrize(
    "backend, reaches",
    [("remote", None), ("cpu", "cpu"), (None, None)],
)
def test_a_teacher_test_never_loads_a_model_on_a_cluster(
    taught, monkeypatch, backend, reaches
):
    """A preview runs here. The forms that open it can target a cluster and hand over the
    *run's* target, where `remote` names where the distill is submitted and is not a device
    (`tests/api/test_preview_compute.py`: the prompt test hit this first). What reaches the
    dataset job, and so the model's device, is the local target or nothing."""
    from detail.core.dataclasses import ComputeTarget

    it, _recording = taught
    seen = []

    class Reached(Exception):
        pass

    def stop_at_the_job(root, job, report):
        seen.append(job.compute)
        raise Reached

    monkeypatch.setattr(preview.kds, "_gather", stop_at_the_job)
    target = (
        None
        if backend is None
        else ComputeTarget(backend, None, None, None, "tillicum")
    )
    with pytest.raises(Reached):
        preview.run_teacher_test(it.experiment.dir, DETECT, compute=target, n_frames=3)

    assert (seen[0].backend if seen[0] else None) == reaches


def test_the_cli_tests_a_teacher_and_writes_nothing(taught):
    from click.testing import CliRunner

    from detail.cli import main

    it, _recording = taught
    before = sorted(p.name for p in it.experiment.dir.rglob("*"))
    result = CliRunner().invoke(
        main,
        [
            "keypoints",
            "test",
            "--experiment",
            str(it.experiment.dir),
            "--teacher",
            "detect",
            "--recording",
            "recording_002",
            "--n",
            "2",
            "--cutoff",
            "0.4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "fake_kp on 2 of 2 frame(s)" in result.output
    assert "scores are heatmap peak, 0 to 1" in result.output
    assert (
        "cut-off 0.4:" in result.output and "would be taught visible" in result.output
    )
    assert "held-out check: not available" in result.output
    assert sorted(p.name for p in it.experiment.dir.rglob("*")) == before


def test_the_cli_test_draws_the_way_the_distill_would(taught, monkeypatch):
    """`--every-n-frames` / `--exclude-audit-s` reach the test, as the GUI's Test cut-off…
    passes its form's: a test drawn differently from the distill is not a sample of it.
    """
    from click.testing import CliRunner

    from detail.api.experiment import Experiment
    from detail.cli import main

    it, _recording = taught
    seen = {}
    real = Experiment.test_keypoints

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Experiment, "test_keypoints", spy)
    result = CliRunner().invoke(
        main,
        [
            "keypoints",
            "test",
            "--experiment",
            str(it.experiment.dir),
            "--teacher",
            "detect",
            "--recording",
            "recording_002",
            "--n",
            "1",
            "--every-n-frames",
            "2",
            "--exclude-audit-s",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["every_n_frames"] == 2 and seen["exclude_audit_s"] == 0.5


def test_the_phases_say_whether_it_waits_on_frames_or_the_model(patch_teacher):
    """Reported from the app: Test cut-off sat on "loading" with the work apparently done. The
    bar was driven by frames *queued* for the model, so it reached 100% and then the whole batch
    ran unreported. Each phase is named, and "placing points" counts what the model returned.
    """
    it, _recording, set_id, _seen = patch_teacher
    from detail.api import keypoint_distill

    teacher, compute = keypoint_distill.resolve_teacher(
        it.experiment,
        example_set=set_id,
        tracking=["recording_002"],
        model_id="fake_learner",
    )
    events = []
    preview.run_teacher_test(
        it.experiment.dir,
        teacher,
        compute=compute,
        n_frames=4,
        progress=lambda stage, fraction, detail: events.append(
            (stage, fraction, detail)
        ),
    )
    stages = [s for s, _f, _d in events]
    firsts = [s for i, s in enumerate(stages) if s not in stages[:i]]
    assert firsts[:4] == [
        "reading tracked frames",
        "loading fake_learner",
        "decoding frames",
        "placing points",
    ], firsts
    placing = [(i, f) for i, (s, f, _d) in enumerate(events) if s == "placing points"]
    assert placing[0][1] < 1.0 and placing[-1][1] == 1.0
    last_decode = max(i for i, s in enumerate(stages) if s == "decoding frames")
    full = [i for i, f in placing if f == 1.0]
    assert min(full) > last_decode, "100% before the model had run"
    assert "on cpu" in events[full[0]][2], "says where the model runs"


def test_the_patch_teacher_carries_the_detect_forms_method_settings(patch_teacher):
    """*Test keypoints* on the detect form places points the way the run will: the form's
    decode / left-right rule / crop size reach the method the test builds, and a setting
    left out is the method's own default, as on a detect run."""
    it, _recording, set_id, _seen = patch_teacher
    from detail.api import keypoint_distill

    teacher, _compute = keypoint_distill.resolve_teacher(
        it.experiment,
        example_set=set_id,
        tracking=["recording_002"],
        model_id="fake_learner",
        settings={"decode": "per_point", "crop_px": 96},
    )
    assert teacher["settings"]["decode"] == "per_point"
    assert teacher["settings"]["crop_px"] == 96
    with pytest.raises(ValueError, match="unknown method setting"):
        keypoint_distill.resolve_teacher(
            it.experiment,
            example_set=set_id,
            tracking=["recording_002"],
            model_id="fake_learner",
            settings={"batch": 4},
        )
    with pytest.raises(ValueError, match="method settings are for the patch-method"):
        keypoint_distill.resolve_teacher(
            it.experiment, DETECT["detect_runs"], settings={"crop_px": 96}
        )


def test_the_cli_test_takes_the_detect_methods_settings(patch_teacher, monkeypatch):
    """`--decode` / `--side-rule` / `--crop-px` reach the patch teacher, as the detect
    form's Test keypoints passes its Advanced settings; a detect-run teacher refuses them.
    """
    from click.testing import CliRunner

    from detail.api.experiment import Experiment
    from detail.cli import main

    it, _recording, set_id, _seen = patch_teacher
    seen = {}
    monkeypatch.setattr(
        Experiment,
        "test_keypoints",
        lambda self, **kwargs: seen.update(kwargs)
        or (_ for _ in ()).throw(ValueError("stop here")),
    )
    base = ["keypoints", "test", "--experiment", str(it.experiment.dir)]
    patch = [
        "--teacher",
        "patch",
        "--example-set",
        set_id,
        "--tracking-run",
        "recording_002:track_01",
        "--model",
        "fake_learner",
    ]
    result = CliRunner().invoke(
        main, base + patch + ["--crop-px", "96", "--decode", "per_point"]
    )
    assert "stop here" in result.output, result.output
    assert seen["settings"] == {"crop_px": 96, "decode": "per_point"}

    seen.clear()
    CliRunner().invoke(main, base + patch)
    assert "settings" not in seen, "nothing typed, nothing sent: the method's defaults"

    result = CliRunner().invoke(
        main,
        base
        + ["--teacher", "detect", "--recording", "recording_002", "--crop-px", "96"],
    )
    assert result.exit_code != 0 and "are for --teacher patch" in result.output
