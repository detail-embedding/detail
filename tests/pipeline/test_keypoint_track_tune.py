"""``keypoint_track_tune`` (§26 *Tuning a track method*): every grid setting scored against an
example set's labels, the choice made on one half of the labelled recordings and checked on the
other, and the detect run's own example set refused.

Two recordings with a real detect run each (the fake method from ``test_keypoint_detect``, on
disk), and a labelled example set over both.
"""

import dataclasses

import cv2
import numpy as np
import pytest

from detail.configs.keypoint_track import TemporalDecodeConfig, tune_settings
from detail.core.examples import ExampleFrame, ExampleMark
from detail.core.registries import KEYPOINT_MODELS
from detail.core.skeleton import Skeleton
from detail.pipeline import keypoint_track_tune as tune
from detail.storage import example_sets
from detail.storage import keypoint_runs
from detail.storage import recording as recording_store
from detail.storage.behavior_audit import MIN_N_FOR_ESTIMATE
from detail.storage.zarr_io import TracksSummaryWriter
from tests.pipeline.test_keypoint_detect import (
    BOX_A,
    SKELETON,
    _FakeKeypointModel,
    _config,
    _run,
)

N_FRAMES = MIN_N_FOR_ESTIMATE + 2  # each half one recording, past the scorer's minimum


def _recording(it, recording_id):
    sq = it.square
    detections = [(f, 0, BOX_A, sq(4, 20, 4, 20)) for f in range(N_FRAMES)]
    recording = it.add_recording(recording_id, detections, [0] * N_FRAMES)
    video = recording.dir / "video" / "part_001.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (40, 40)
    )
    for i in range(N_FRAMES):
        writer.write(np.full((40, 40, 3), 20 * i, np.uint8))
    writer.release()
    with TracksSummaryWriter(
        recording.dir / "tracking" / "track_01" / "tracks.zarr"
    ) as w:
        w.add_track(0, "mouse", 0, N_FRAMES - 1, N_FRAMES, 0, 0.9)
    _run(it, recording, _config())
    data = recording_store.read_recording(recording.dir)
    data["keypoint_detect_runs"] = [
        {"id": "kd_01", "status": "complete", "model_id": "fake_kp", "output": "x"}
    ]
    recording_store.write_recording(recording.dir, data)
    return recording


def _labels(it, recordings, name="test labels", *, swap_in=None):
    """An example set labelling every frame of track 0: the nose at the box's top-left corner
    (where the fake method puts it) and the tail at the bottom-right. ``swap_in`` labels that
    recording's frames with the two points exchanged."""
    labels = example_sets.create_set(it.experiment.dir, name, SKELETON)
    for recording in recordings:
        # one detection per frame, so one instance per frame
        rows = keypoint_runs.KeypointDetectRun(recording.dir, "kd_01").rows("part_001")
        for f, t_ns in zip(rows["frame_idx"].tolist(), rows["t_ns"].tolist()):
            nose, tail = (BOX_A[0], BOX_A[1]), (BOX_A[2], BOX_A[3])
            if recording.id == swap_in:
                nose, tail = tail, nose
            labels.frames.append(
                ExampleFrame(
                    image=f"frames/{recording.id}_{f}.png",
                    recording_id=recording.id,
                    tracking_run_id="track_01",
                    track_id=0,
                    part_id="part_001",
                    frame_idx=int(f),
                    t_ns=int(t_ns),
                    bbox=BOX_A,
                    marks=[
                        ExampleMark("nose", x=nose[0], y=nose[1]),
                        ExampleMark("tail", x=tail[0], y=tail[1]),
                    ],
                )
            )
    example_sets.write_set(it.experiment.dir, labels)
    return labels


@pytest.fixture
def two_recordings(processed_recording, monkeypatch):
    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp", _FakeKeypointModel)
    it = processed_recording
    return it, [_recording(it, "recording_002"), _recording(it, "recording_003")]


def _detect_runs(recordings) -> dict:
    return {r.id: "kd_01" for r in recordings}


#: The tracking run whose subject the instances are assigned to (§26 *Assignment*), in every
#: recording the fixture builds.
TRACKING = {"recording_002": "track_01", "recording_003": "track_01"}


def test_every_grid_setting_is_a_valid_temporal_decode_config():
    settings = tune_settings()
    assert len(settings) == len({tune.setting_name(s) for s in settings})
    for setting in settings:
        TemporalDecodeConfig.model_validate(setting)


def test_tuning_scores_detection_the_current_settings_and_every_grid_setting(
    two_recordings,
):
    it, recordings = two_recordings
    labels = _labels(it, recordings)
    events = []
    result = tune.tune(
        it.experiment.dir,
        _detect_runs(recordings),
        labels.id,
        progress=lambda *e: events.append(e),
        tracking_runs=TRACKING,
    )
    names = [s.name for s in result.settings]
    assert names[:2] == [tune.DETECTION_ONLY, tune.CURRENT]
    assert len(names) == 2 + len(tune_settings())
    assert result.coverage == {
        "labelled": 2 * N_FRAMES,
        "excluded": 0,
        "uncovered": 0,
        "scored": 2 * N_FRAMES,
    }
    assert result.checked and len(result.pick_recordings) == 1
    assert sorted(result.pick_recordings + result.check_recordings) == [
        "recording_002",
        "recording_003",
    ]
    detection = result.setting(tune.DETECTION_ONLY)
    assert detection.figure("all", "group", "all")["value"] == pytest.approx(1.0)
    assert detection.figure("check", "keypoint", "nose")["value"] == pytest.approx(1.0)
    assert result.best in names[2:]
    TemporalDecodeConfig.model_validate(result.best_config())
    assert {e[0] for e in events} >= {"tracking", "scoring"}
    assert events[-1][1] == 1.0


def _scored(name, overrides, pick, check):
    setting = tune.TuneSetting(name, overrides)
    setting.scores = {
        "pick": {"group:all": {"pck10": {"value": pick}}},
        "check": {"group:all": {"pck10": {"value": check}}},
        "all": {"group:all": {"pck10": {"value": (pick + check) / 2}}},
    }
    return setting


def test_the_choice_reads_the_pick_half_and_never_the_check_half():
    """The guard for this module's class (labels choose a setting; the check half must not):
    the check half and the pooled figures point at another setting than the pick half, so a
    choice that read either fails here. Detection alone and the current settings are never
    chosen, however they score."""
    settings = [
        _scored(tune.DETECTION_ONLY, None, 0.99, 0.99),
        _scored(tune.CURRENT, {}, 0.98, 0.98),
        _scored("a", {"transition_sigma_body": 1.0}, 0.70, 0.95),
        _scored("b", {"transition_sigma_body": 2.0}, 0.80, 0.40),
        _scored("c", {"transition_sigma_body": 3.0}, 0.80, 0.10),
    ]
    assert (
        tune.choose(settings).name == "b"
    )  # first of the pick-half tie, in grid order
    settings[3].scores["pick"]["group:all"]["pck10"]["value"] = None  # withheld
    assert tune.choose(settings).name == "c"


def _per_point(name, overrides, pick: dict, check: dict):
    setting = tune.TuneSetting(name, overrides)
    setting.scores = {
        half: {f"keypoint:{p}": {"pck10": {"value": v}} for p, v in values.items()}
        for half, values in (("pick", pick), ("check", check))
    }
    return setting


def test_points_where_detection_wins_on_the_pick_half_are_left_as_detected():
    """Any skeleton: a pair counts as one unit (both points, mean), a single point alone, and
    only the pick half is read. The check half here says the opposite at every point; a
    choice that read it would keep different points."""
    skeleton = Skeleton(
        names=("snout", "l_fin", "r_fin", "tail"),
        pairs=(("l_fin", "r_fin"),),
    )
    tracked = list(skeleton.names)
    best = _per_point(
        "best",
        {"transition_sigma_body": 3.0},
        {"snout": 0.9, "l_fin": 0.4, "r_fin": 0.6, "tail": 0.8},
        {"snout": 0.1, "l_fin": 0.9, "r_fin": 0.9, "tail": 0.1},
    )
    detection = _per_point(
        tune.DETECTION_ONLY,
        None,
        {"snout": 0.7, "l_fin": 0.7, "r_fin": 0.6, "tail": 0.8},
        {"snout": 0.9, "l_fin": 0.1, "r_fin": 0.1, "tail": 0.9},
    )
    # snout tracked (0.9 > 0.7); the fins as a pair (0.5 < 0.65) left; tail tie stays tracked
    assert tune.points_to_track(best, detection, skeleton, tracked) == [
        "snout",
        "tail",
    ]
    # a pair only half tracked is decided point by point
    assert tune.points_to_track(
        best, detection, skeleton, ["snout", "l_fin", "tail"]
    ) == ["snout", "tail"]
    # every unit worth tracking: nothing to change
    detection.scores["pick"] = {
        f"keypoint:{p}": {"pck10": {"value": 0.0}} for p in skeleton.names
    }
    assert tune.points_to_track(best, detection, skeleton, tracked) is None
    # a withheld figure keeps its unit tracked, for want of evidence
    detection.scores["pick"]["keypoint:snout"]["pck10"]["value"] = 1.0
    best.scores["pick"]["keypoint:snout"]["pck10"]["value"] = None
    assert tune.points_to_track(best, detection, skeleton, tracked) is None


def test_the_choice_is_made_on_the_pick_half_alone(two_recordings):
    """End to end: scrambling the check half's labels changes its figures, never ``best``.
    (Every setting scores alike over the fake method, so this exercises the wiring; the guard
    with teeth is the test above.)"""
    it, recordings = two_recordings
    plain = tune.tune(
        it.experiment.dir,
        _detect_runs(recordings),
        _labels(it, recordings).id,
        tracking_runs=TRACKING,
    )
    check = plain.check_recordings[0]
    scrambled = tune.tune(
        it.experiment.dir,
        _detect_runs(recordings),
        _labels(it, recordings, "scrambled", swap_in=check).id,
        tracking_runs=TRACKING,
    )
    assert scrambled.check_recordings == plain.check_recordings
    assert scrambled.best == plain.best
    before = plain.setting(plain.best).figure("check", "group", "all")["value"]
    after = scrambled.setting(plain.best).figure("check", "group", "all")["value"]
    assert after < before


@pytest.mark.parametrize("per_class", [False, True], ids=["one_set", "per_class"])
def test_the_detect_runs_own_example_set_is_refused(two_recordings, per_class):
    """Any set the run learned from is refused — for a per-class run, the second class's set
    too, which a check reading the run's one ``example_set`` would miss (§26 *One example set
    per detection class*)."""
    it, recordings = two_recordings
    labels = _labels(it, recordings)
    run_dir = keypoint_runs.detect_run_dir(recordings[0].dir, "kd_01")
    if per_class:
        keypoint_runs.update_detect_yaml(
            run_dir,
            example_set=None,
            example_sets={
                "Intruder": {"id": "ex_99", "marks": []},
                "Resident": {"id": labels.id, "marks": []},
            },
        )
    else:
        keypoint_runs.update_detect_yaml(run_dir, example_set={"id": labels.id})
    with pytest.raises(ValueError, match="answer key"):
        tune.tune(
            it.experiment.dir,
            _detect_runs(recordings),
            labels.id,
            tracking_runs=TRACKING,
        )


def test_labels_without_a_complete_detect_run_are_counted_and_none_left_refuses(
    two_recordings,
):
    it, recordings = two_recordings
    labels = _labels(it, recordings)
    result = tune.tune(
        it.experiment.dir,
        {recordings[0].id: "kd_01"},
        labels.id,
        tracking_runs=TRACKING,
    )
    assert result.coverage["uncovered"] == N_FRAMES
    assert not result.checked  # one recording: nothing to check the choice on
    with pytest.raises(ValueError, match="no labelled frame"):
        tune.tune(it.experiment.dir, {}, labels.id, tracking_runs=TRACKING)


@pytest.mark.parametrize("per_class", [False, True], ids=["one_set", "per_class"])
def test_the_sets_offered_leave_out_the_detect_runs_own_and_another_skeletons(
    two_recordings, per_class
):
    """By the detect spec's own digests: the set it learned from (its content digest, or any
    class's for a per-class run) and a set labelled on another skeleton are not usable, and
    say why; an unlabelled set is not listed at all."""
    it, recordings = two_recordings
    own = _labels(it, recordings, "the run's own")
    other = _labels(it, recordings, "held out", swap_in=recordings[1].id)
    fish = example_sets.create_set(
        it.experiment.dir,
        "fish",
        Skeleton(names=("snout", "caudal"), edges=(("snout", "caudal"),)),
    )
    fish.frames = [dataclasses.replace(f, marks=[]) for f in other.frames[:1]]
    fish.frames[0].marks = [ExampleMark("snout", x=0.2, y=0.2)]
    fish.frames[0].marks.append(ExampleMark("caudal", x=0.4, y=0.4))
    example_sets.write_set(it.experiment.dir, fish)
    example_sets.create_set(it.experiment.dir, "empty", SKELETON)
    own_digest = own.digest(example_sets.image_hashes(it.experiment.dir, own))
    learned = (
        {
            "example_set_digest": None,
            "class_example_digests": {"Intruder": "0" * 16, "Resident": own_digest},
        }
        if per_class
        else {"example_set_digest": own_digest}
    )
    spec = {
        "module": "keypoint_detect",
        "identity": {**learned, "skeleton_digest": SKELETON.digest()},
    }
    choices = {c["id"]: c for c in tune.example_set_choices(it.experiment.dir, spec)}
    assert set(choices) == {
        own.id,
        other.id,
        fish.id,
    }  # the unlabelled set is not listed
    assert choices[own.id]["reason"] == "the detect run learned from it"
    assert choices[other.id]["usable"] and choices[other.id]["reason"] is None
    assert "another skeleton" in choices[fish.id]["reason"]
    assert choices[other.id]["labelled"] == 2 * N_FRAMES
    everything = tune.example_set_choices(it.experiment.dir, None)  # no detect run yet
    assert all(c["usable"] for c in everything)


def test_split_is_seeded_balanced_and_needs_two_recordings():
    assert tune.split_recordings({"a": 5}) == (["a"], [])
    counts = {"a": 10, "b": 1, "c": 1, "d": 9}
    pick, check = tune.split_recordings(counts, seed=3)
    assert tune.split_recordings(counts, seed=3) == (pick, check)
    assert sorted(pick + check) == sorted(counts)
    assert abs(sum(counts[r] for r in pick) - sum(counts[r] for r in check)) <= 1
    tied = {r: 4 for r in "abcdef"}
    assert {tuple(tune.split_recordings(tied, seed=s)[0]) for s in range(6)} != {
        tuple(tune.split_recordings(tied, seed=0)[0])
    }  # the seed decides among equal counts
