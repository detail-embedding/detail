"""The keypoint setup run (§26, #717): a teacher across acquisition setups, then its student.

One distill run of kind ``keypoint_setup``: detect and track one short window of each setup,
stop at ``awaiting_references`` for a person to click an example set, then teach with the
patch method on those tracks and distill as a ``keypoints`` distill does (label → train →
register). Ported from ``remote-detail`` onto main's classes in the 2026-10-16 catch-up: the
reference set became an example set, and rd's "place the teacher's keypoints" processing run
is gone, because main's patch teacher labels the sampled frames itself.

The teacher stages are processing runs and the student stages are tested elsewhere
(``tests/api/test_keypoint_distill.py``), so here they are stand-ins, and what is pinned is the
plan, the up-front refusals, the pause, what each stage is handed, and that the run continues
past the pause without redoing what it already did.

A stand-in still checks what it is handed against the real module's config model
(``_as_the_module_reads_it``). Without that, a stage config the module refuses passes here: the
setup handed tracking ``tracker_id: None`` and every real setup failed at its tracking stage
while this file stayed green (``tests/configs/test_none_means_default.py``).
"""

import types
import typing

import pydantic
import pytest

from detail.api import keypoint_distill
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.api.experiment import DistillOrchestrator, Experiment
from detail.core.dataclasses import (
    AWAITING_REFERENCES,
    KEYPOINT_SETUP_CLASSES_STAGE_WEIGHTS,
    KEYPOINT_SETUP_STAGE_WEIGHTS,
    ComputeTarget,
    LabelingProgressEvent,
    ProcessingProgress,
    TrainingProgressEvent,
    distill_overall_fraction,
)
from detail.core.registries import TRAINERS
from detail.executor.runners import PROCESSING_MODULES, import_runner_modules
from detail.pipeline import teacher_window
from detail.storage import distill_runs as distill_store
from detail.storage import recording as recording_store
from detail.storage import runs as runs_store
from detail.storage.metadata import ACQUISITION_SETUP_KEY as SETUP

CPU = ComputeTarget("cpu", None, None, None)


class _FakeTrainer:
    supported_backends = ["cpu", "cuda"]

    def is_ready(self):
        return True

    def validate_hyperparams(self, hyperparams):
        return dict(hyperparams)


class _FakeDetector:
    supported_cue_types = ()  # closed-set: a distilled detector
    supported_backends = ["cpu", "cuda"]
    classes = ["mouse", "hand"]


@pytest.fixture
def setup_env(monkeypatch):
    monkeypatch.setitem(TRAINERS, "faketrainer", _FakeTrainer)
    monkeypatch.setattr(
        "detail.models.resolve.resolve_detection_model_auto",
        lambda root, model_id, variant=None: (_FakeDetector(), "trained"),
    )
    spawned = []
    monkeypatch.setattr(
        "detail.api.experiment.spawn_coordinator",
        lambda run_dir, *a, **k: spawned.append(run_dir) or 4242,
    )
    return spawned


def _ready_set(monkeypatch):
    """An example set that passes the checks, without clicking one."""
    monkeypatch.setattr(keypoint_distill, "_check_example_set", lambda exp, s: None)


def _recording(experiment, name, setup, duration_s=600.0, video=True):
    recording = experiment.add_recording(name)
    data = recording_store.read_recording(recording.dir)
    data["video"] = {
        "files": [{"index": 0, "file": f"video/{name}.mp4"}] if video else [],
        "total_duration_s": duration_s,
    }
    recording_store.write_recording(recording.dir, data)
    experiment.set_recording_metadata(recording.id, {SETUP: setup})
    return recording


def _three_setups(tmp_path):
    exp = Experiment.create("p", tmp_path)
    a1 = _recording(exp, "box1_day1", "box 1", duration_s=1800)
    _recording(exp, "box2_day1", "box 2", duration_s=120)  # shorter than the window
    _recording(exp, "box1_day2", "box 1")
    _recording(exp, "top_cam", "top camera")
    return exp, a1


def _start(exp, **over):
    args = dict(
        name="rat pose",
        detector_model_id="model_001",
        class_name="mouse",
        detect_compute=CPU,
        keypoints_compute=CPU,
        train_compute=CPU,
        trainer_id="faketrainer",
    )
    args.update(over)
    return exp.start_keypoint_setup(**args)


# --- the plan ---------------------------------------------------------------------------- #
def test_the_plan_is_the_first_recording_of_each_setup(tmp_path):
    exp, a1 = _three_setups(tmp_path)
    plan = exp.keypoint_teacher_plan(window_s=300)
    assert [row["acquisition_setup"] for row in plan] == [
        "box 1",
        "box 2",
        "top camera",
    ]
    first = plan[0]
    assert first["recording_id"] == a1.id and first["windowed"]
    assert len(first["recordings"]) == 2  # the other box 1 recording, for a swap
    assert not plan[1]["windowed"]  # 120 s is shorter than the window


def test_the_plan_refuses_a_blank_setup_and_names_it(tmp_path):
    exp, _ = _three_setups(tmp_path)
    _recording(exp, "mystery", "")
    with pytest.raises(ValueError, match="'mystery'"):
        exp.keypoint_teacher_plan()


def test_the_plan_skips_recordings_without_video(tmp_path):
    exp, _ = _three_setups(tmp_path)
    _recording(exp, "no_video", "", video=False)  # blank, but skipped: no video
    assert len(exp.keypoint_teacher_plan()) == 3


# --- starting ---------------------------------------------------------------------------- #
def test_starting_reserves_a_setup_run_over_one_recording_per_setup(
    tmp_path, setup_env
):
    exp, a1 = _three_setups(tmp_path)
    run = _start(exp, window_s=240, score_cutoff=0.4)
    record = distill_store.read_run(exp.dir, run.id)
    assert record["kind"] == "keypoint_setup" and record["status"] == "running"
    cfg = record["config"]
    assert len(cfg["recording_ids"]) == 3 and cfg["recording_ids"][0] == a1.id
    assert cfg["window_s"] == 240.0 and cfg["example_set_id"] is None
    assert cfg["keypoint_dataset"]["score_cutoff"] == 0.4
    assert list(record["stages"]) == list(keypoint_distill.SETUP_STAGES)
    assert len(setup_env) == 1  # the coordinator was spawned


def test_a_teacher_can_be_swapped_for_another_of_its_setup(tmp_path, setup_env):
    exp, _ = _three_setups(tmp_path)
    run = _start(exp, teachers={"box 1": "box1_day2"})
    cfg = distill_store.read_run(exp.dir, run.id)["config"]
    assert exp.get_recording(cfg["teachers"]["box 1"]).name == "box1_day2"
    with pytest.raises(ValueError, match="not one of setup 'box 1'"):
        _start(exp, teachers={"box 1": "top_cam"})


@pytest.mark.parametrize(
    "over, fragment",
    [
        ({"name": " "}, "give the keypoint setup a name"),
        ({"class_name": "rat"}, "has no class 'rat'"),
        ({"tracker_id": "nope"}, "unknown tracker_id"),
        ({"teachers": {"box 9": "box1_day1"}}, "no acquisition setup 'box 9'"),
        ({"base_model": "yolo11s-seg.pt"}, "not a pose checkpoint"),
        ({"window_s": 0}, "window_s must be > 0"),
    ],
)
def test_starting_refuses_what_cannot_work(tmp_path, setup_env, over, fragment):
    exp, _ = _three_setups(tmp_path)
    with pytest.raises(ValueError, match=fragment):
        _start(exp, **over)
    assert distill_store.list_runs(exp.dir) == []  # nothing reserved


def test_starting_refuses_a_prompted_detector(tmp_path, setup_env, monkeypatch):
    exp, _ = _three_setups(tmp_path)
    monkeypatch.setattr(
        "detail.models.resolve.resolve_detection_model_auto",
        lambda root, model_id, variant=None: (
            types.SimpleNamespace(supported_cue_types=[object]),
            "pretrained",
        ),
    )
    with pytest.raises(ValueError, match="is a prompted model"):
        _start(exp)


def test_starting_refuses_a_method_that_learns_from_no_examples(tmp_path, setup_env):
    exp, _ = _three_setups(tmp_path)
    with pytest.raises(ValueError, match="without an example set"):
        _start(exp, keypoints_model_id="superanimal_quadruped")


def test_a_set_given_at_start_needs_a_cut_off_dinov3_match_has_none(
    tmp_path, setup_env, monkeypatch
):
    exp, _ = _three_setups(tmp_path)
    _ready_set(monkeypatch)
    with pytest.raises(ValueError, match="sets no default score cut-off"):
        _start(exp, example_set="ex_01")
    assert distill_store.list_runs(exp.dir) == []


# --- the orchestrator: teacher, pause, continue ------------------------------------------ #
def _as_the_module_reads_it(module_id, config):
    """Raise unless every setting in ``config`` that ``module_id``'s config model declares is a
    value of that field's type and bounds, as handed over (no defaults filled in) — and unless
    a stage after detection is one that takes a teacher run.

    The setup's detection is windowed, so everything after it reads a teacher run. A stage that
    refuses one (``teacher_window.refuse_teacher_run``) fails every real setup at that stage
    while a stand-in passes: the two-animal merge did, with "merging tracks needs a run over the
    whole recording" (``tests/pipeline/test_teacher_window.py``)."""
    import_runner_modules(PIPELINE_RUNNERS)
    assert (
        module_id == "detection" or module_id in teacher_window.TEACHER_CONSUMERS
    ), f"the setup hands {module_id!r} a teacher run, which it refuses"
    fields = PROCESSING_MODULES[module_id].config_model.model_fields
    for key, value in config.items():
        if key in fields:
            field = fields[key]
            annotation = field.annotation
            if field.metadata:
                annotation = typing.Annotated[(annotation, *field.metadata)]
            pydantic.TypeAdapter(annotation).validate_python(value)


class _FakeProcessingRun:
    """A finished processing run over ``recording_ids``, recorded where the stage reads it."""

    def __init__(self, exp, run_id, recording_ids, prefix):
        self.id, self.status, self.error = run_id, "complete", None
        self._n = len(recording_ids)
        runs_store.write_run(
            exp.dir,
            run_id,
            {
                "id": run_id,
                "status": "complete",
                "per_recording": {
                    sid: {"run_id": f"{prefix}_{i + 1:02d}", "status": "complete"}
                    for i, sid in enumerate(recording_ids)
                },
            },
        )

    def progress(self):
        return iter([ProcessingProgress(1, self._n, "r", None)])

    def wait(self):
        return self


def _stubbed_stages(exp, monkeypatch):
    calls = {
        "processing": [],
        "dataset": [],
        "train": [],
        "register": [],
        "teacher": [],
    }
    prefixes = {"detection": "det", "tracking": "trk"}

    def run_processing(module_id, recording_ids, config):
        _as_the_module_reads_it(module_id, config)
        calls["processing"].append((module_id, config))
        run_id = f"run_{len(calls['processing']):03d}"
        return _FakeProcessingRun(exp, run_id, recording_ids, prefixes[module_id])

    def dataset(**kw):
        calls["dataset"].append(kw)
        return types.SimpleNamespace(
            id="dataset_001",
            status="complete",
            error=None,
            progress=lambda: iter([LabelingProgressEvent(5, 5, 0, 1, 1.0, 0.0, None)]),
            wait=lambda: None,
        )

    def train(**kw):
        calls["train"].append(kw)
        return types.SimpleNamespace(
            id="model_002",
            status="complete",
            error=None,
            progress=lambda: iter([TrainingProgressEvent(5, 5, {}, 1.0, 0.0, None)]),
            wait=lambda: None,
        )

    def resolve_teacher(experiment, example_set, tracking, *rest):
        calls["teacher"].append((example_set, dict(tracking)))
        return {"kind": "model", "model_id": "dinov3_match"}, CPU

    monkeypatch.setattr(exp, "run_processing", run_processing)
    monkeypatch.setattr(exp, "keypoint_dataset", dataset)
    monkeypatch.setattr(exp, "train_model", train)
    monkeypatch.setattr(keypoint_distill, "_resolve_patch_teacher", resolve_teacher)
    monkeypatch.setattr(
        keypoint_distill,
        "register_student",
        lambda experiment, run_id, record: calls["register"].append(run_id)
        or {"name": "rat pose distill_001"},
    )
    monkeypatch.setattr(
        "detail.api.run_spec.spec_from_run",
        lambda recording, module_id, run_id: {"module": module_id, "from": run_id},
    )
    return calls


def _drive(exp, run_id):
    events = []
    record = distill_store.read_run(exp.dir, run_id)
    DistillOrchestrator(exp, run_id, record, events.append).run()
    return distill_store.read_run(exp.dir, run_id), events


def test_the_teacher_stops_for_example_clicks_then_carries_on(
    tmp_path, setup_env, monkeypatch
):
    exp, a1 = _three_setups(tmp_path)
    run = _start(exp, window_s=300)
    calls = _stubbed_stages(exp, monkeypatch)

    record, events = _drive(exp, run.id)
    assert record["status"] == AWAITING_REFERENCES
    assert record["stages"]["examples"]["status"] == AWAITING_REFERENCES
    (det_module, det_cfg), (trk_module, trk_cfg) = calls["processing"]
    assert det_module == "detection" and det_cfg["teacher_window_s"] == 300.0
    assert det_cfg["model_id"] == "model_001" and det_cfg["prompt"].targets == []
    assert trk_module == "tracking" and trk_cfg["class_name"] == "mouse"
    assert trk_cfg["detection_spec"] == {"module": "detection", "from": "det_01"}
    track_runs = record["stages"]["track"]["run_ids"]
    assert track_runs[a1.id] == "trk_01"  # where the person clicks their examples
    assert not calls["dataset"]  # nothing past the pause ran

    _ready_set(monkeypatch)
    exp.continue_keypoint_setup(run.id, "ex_01", score_cutoff=0.4)
    record, events = _drive(exp, run.id)
    assert record["status"] == "complete"
    assert [module for module, _ in calls["processing"]] == ["detection", "tracking"]
    # the patch teacher, built from the set on the teacher's own tracks
    assert calls["teacher"] == [("ex_01", track_runs)]
    (dataset,) = calls["dataset"]
    assert dataset["teacher"]["model_id"] == "dinov3_match"
    assert dataset["score_cutoff"] == 0.4
    assert calls["register"] == [run.id]
    assert {e.stage_count for e in events} == {6}
    fractions = [e.overall_fraction for e in events]
    assert fractions == sorted(fractions)  # the bar never runs backwards


def test_a_set_given_at_start_runs_straight_through(tmp_path, setup_env, monkeypatch):
    exp, _ = _three_setups(tmp_path)
    _ready_set(monkeypatch)
    run = _start(exp, example_set="ex_01", score_cutoff=0.4)
    calls = _stubbed_stages(exp, monkeypatch)
    record, _ = _drive(exp, run.id)
    assert record["status"] == "complete"  # never stopped for clicks
    examples = record["stages"]["examples"]
    assert (examples["status"], examples["example_set_id"]) == ("complete", "ex_01")
    assert calls["teacher"][0][0] == "ex_01"


def test_continue_needs_a_run_that_is_waiting(tmp_path, setup_env):
    exp, _ = _three_setups(tmp_path)
    run = _start(exp)
    with pytest.raises(ValueError, match="only a keypoint setup waiting"):
        exp.continue_keypoint_setup(run.id, "ex_01")


def test_continue_needs_a_cut_off_when_the_method_has_none(
    tmp_path, setup_env, monkeypatch
):
    exp, _ = _three_setups(tmp_path)
    run = _start(exp)
    _stubbed_stages(exp, monkeypatch)
    _drive(exp, run.id)
    _ready_set(monkeypatch)
    with pytest.raises(ValueError, match="sets no default score cut-off"):
        exp.continue_keypoint_setup(run.id, "ex_01")
    assert distill_store.read_run(exp.dir, run.id)["status"] == AWAITING_REFERENCES


def test_a_waiting_setup_is_continued_not_resumed_and_can_be_cancelled(
    tmp_path, setup_env, monkeypatch
):
    exp, _ = _three_setups(tmp_path)
    run = _start(exp)
    _stubbed_stages(exp, monkeypatch)
    _drive(exp, run.id)
    with pytest.raises(ValueError, match="continue_keypoint_setup"):
        exp.resume_distill(run.id)
    exp.cancel_distill(run.id)
    assert distill_store.read_run(exp.dir, run.id)["status"] == "cancelled"


def test_the_setup_bar_folds_its_stages_in_order():
    weights = KEYPOINT_SETUP_STAGE_WEIGHTS
    assert sum(weights.values()) == pytest.approx(1.0)
    assert distill_overall_fraction("detecting", 0.0, weights) == 0.0
    assert distill_overall_fraction("tracking", 0.0, weights) == pytest.approx(0.25)
    assert distill_overall_fraction("training", 1.0, weights) == pytest.approx(1.0)
    assert distill_overall_fraction("registering", 0.0, weights) == pytest.approx(1.0)


# --- two or more animals: each tracked, then merged (§26 *One example set per class*) ---- #
ANIMALS = ["mouse", "hand"]  # the fake detector's two classes
SETS = {"mouse": "ex_white", "hand": "ex_black"}


def _per_class_stages(exp, monkeypatch):
    """`_stubbed_stages`, with each tracking run named after its class and the merge
    recorded, and the teacher keeping the per-class map it was handed."""
    calls = _stubbed_stages(exp, monkeypatch)

    def run_processing(module_id, recording_ids, config):
        _as_the_module_reads_it(module_id, config)
        calls["processing"].append((module_id, config))
        run_id = f"run_{len(calls['processing']):03d}"
        prefix = {
            "detection": "det",
            "tracking": f"trk_{config.get('class_name')}",
            "track_merge": "mrg",
        }[module_id]
        return _FakeProcessingRun(exp, run_id, recording_ids, prefix)

    def resolve_teacher(experiment, example_set, tracking, *rest):
        calls["teacher"].append((example_set, dict(tracking), rest[-1]))
        return {"kind": "model", "model_id": "dinov3_match"}, CPU

    monkeypatch.setattr(exp, "run_processing", run_processing)
    monkeypatch.setattr(keypoint_distill, "_resolve_patch_teacher", resolve_teacher)
    monkeypatch.setattr(
        keypoint_distill, "_check_class_example_sets", lambda exp, animals, sets: None
    )
    return calls


def test_two_animals_are_tracked_each_merged_then_wait_for_a_set_each(
    tmp_path, setup_env, monkeypatch
):
    """The track stage runs one class-filtered tracking run per animal, the merge is one
    track_merge over each recording's per-class tracks, and the pause waits for one set per
    class; continued, the teacher labels the **merged** tracks with that map."""
    exp, a1 = _three_setups(tmp_path)
    run = _start(exp, class_name=None, class_names=ANIMALS)
    record = distill_store.read_run(exp.dir, run.id)
    assert record["config"]["class_names"] == ANIMALS
    assert record["config"]["class_name"] is None
    assert list(record["stages"])[:4] == ["detect", "track", "merge", "examples"]
    calls = _per_class_stages(exp, monkeypatch)

    record, _events = _drive(exp, run.id)
    assert record["status"] == AWAITING_REFERENCES
    modules = [(m, c.get("class_name")) for m, c in calls["processing"]]
    assert modules == [
        ("detection", None),
        ("tracking", "mouse"),
        ("tracking", "hand"),
        ("track_merge", None),
    ]
    merge_config = calls["processing"][3][1]
    assert merge_config["source_specs"] == [
        {"module": "tracking", "from": "trk_mouse_01"},
        {"module": "tracking", "from": "trk_hand_01"},
    ]
    track = record["stages"]["track"]
    assert track["run_ids_by_class"]["hand"][a1.id] == "trk_hand_01"
    assert set(track["processing_run_ids"]) == set(ANIMALS)  # each one cancellable
    merged = record["stages"]["merge"]["run_ids"]
    assert merged[a1.id] == "mrg_01"

    with pytest.raises(ValueError, match="one example set per class"):
        exp.continue_keypoint_setup(run.id, "ex_white", score_cutoff=0.4)
    exp.continue_keypoint_setup(run.id, score_cutoff=0.4, class_example_sets=SETS)
    record, events = _drive(exp, run.id)
    assert record["status"] == "complete"
    assert calls["teacher"] == [(None, merged, SETS)]  # the merged tracks, a set each
    assert record["stages"]["examples"]["class_example_sets"] == SETS
    assert {e.stage_count for e in events} == {7}
    fractions = [e.overall_fraction for e in events]
    assert fractions == sorted(fractions)
    # nothing ran twice: the relaunch reused detection, both tracks and the merge
    assert [m for m, _ in calls["processing"]] == [
        "detection",
        "tracking",
        "tracking",
        "track_merge",
    ]


def test_two_animals_with_their_sets_at_start_run_straight_through(
    tmp_path, setup_env, monkeypatch
):
    exp, _ = _three_setups(tmp_path)
    monkeypatch.setattr(
        keypoint_distill, "_check_class_example_sets", lambda exp, animals, sets: None
    )
    run = _start(
        exp,
        class_name=None,
        class_names=ANIMALS,
        class_example_sets=SETS,
        score_cutoff=0.4,
    )
    calls = _per_class_stages(exp, monkeypatch)
    record, _ = _drive(exp, run.id)
    assert record["status"] == "complete"
    assert calls["teacher"][0][2] == SETS


def test_one_class_in_class_names_is_the_one_animal_setup(tmp_path, setup_env):
    exp, _ = _three_setups(tmp_path)
    run = _start(exp, class_name=None, class_names=["mouse"])
    record = distill_store.read_run(exp.dir, run.id)
    assert record["config"]["class_name"] == "mouse"
    assert record["config"]["class_names"] is None
    assert "merge" not in record["stages"]


@pytest.mark.parametrize(
    "over, fragment",
    [
        ({"class_names": ANIMALS}, "name the animal"),  # beside class_name
        ({"class_name": None}, "name the animal"),  # neither
        (
            {"class_name": None, "class_names": ["mouse", "cat"]},
            "has no class 'cat'",
        ),
        ({"class_name": None, "class_names": ["mouse", "mouse"]}, "a class twice"),
        (
            {"class_name": None, "class_names": ANIMALS, "example_set": "ex_01"},
            "one example set per class",
        ),
        (
            {
                "class_name": None,
                "class_names": ANIMALS,
                "class_example_sets": {"mouse": "ex_white"},
            },
            "must name each of",
        ),
        ({"class_example_sets": {"mouse": "ex_white"}}, "for two or more animals"),
    ],
)
def test_starting_refuses_an_animal_list_it_cannot_run(
    tmp_path, setup_env, over, fragment
):
    exp, _ = _three_setups(tmp_path)
    with pytest.raises(ValueError, match=fragment):
        _start(exp, **over)
    assert distill_store.list_runs(exp.dir) == []


def test_a_one_animal_setup_refuses_a_per_class_continue(
    tmp_path, setup_env, monkeypatch
):
    exp, _ = _three_setups(tmp_path)
    run = _start(exp)
    _stubbed_stages(exp, monkeypatch)
    _drive(exp, run.id)
    with pytest.raises(ValueError, match="tracks one animal"):
        exp.continue_keypoint_setup(run.id, score_cutoff=0.4, class_example_sets=SETS)


def test_cancelling_reaches_every_classs_tracking_run(tmp_path, setup_env, monkeypatch):
    """A per-class track stage holds a processing run per class; cancel stops each one,
    not only a single ``processing_run_id`` (the one-class shape)."""
    exp, _ = _three_setups(tmp_path)
    run = _start(exp, class_name=None, class_names=ANIMALS)
    record = distill_store.read_run(exp.dir, run.id)
    record["stages"]["track"] = {
        "status": "running",
        "processing_run_ids": {"mouse": "run_002", "hand": "run_003"},
    }
    distill_store.write_run(exp.dir, run.id, record)
    cancelled = []
    monkeypatch.setattr(exp, "cancel_run", cancelled.append)
    monkeypatch.setattr("detail.api.experiment.kill_coordinator", lambda *a: None)
    exp.cancel_distill(run.id)
    assert sorted(cancelled) == ["run_002", "run_003"]
    stages = distill_store.read_run(exp.dir, run.id)["stages"]
    assert stages["track"]["status"] == "cancelled"


def test_the_per_class_setup_bar_folds_its_stages_in_order():
    weights = KEYPOINT_SETUP_CLASSES_STAGE_WEIGHTS
    assert sum(weights.values()) == pytest.approx(1.0)
    assert list(weights) == ["detecting", "tracking", "merging", "labeling", "training"]
    assert distill_overall_fraction("merging", 0.0, weights) == pytest.approx(0.29)
