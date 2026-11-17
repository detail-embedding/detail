"""The keypoint methods of the Python API (§26, PR 2): example sets and their skeletons,
reached through ``Experiment`` so the delegating one-liners are exercised too."""

import numpy as np
import pytest
import yaml

from detail.api import keypoints as kp
from detail.api.experiment import Experiment
from detail.core.examples import ExampleMark
from detail.core.skeleton import PRESETS, Skeleton
from detail.storage import example_sets as example_store
from detail.storage import recording as recording_store
from tests import experiments

SK = Skeleton(names=("nose", "tail_base"), edges=(("nose", "tail_base"),))
N_FRAMES = 5


@pytest.fixture
def tracked(processed_recording):
    """A second recording with one track on frames 0..4 and a real video."""
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
    built.tracked_recording = recording
    return built


# --- skeleton references ------------------------------------------------------------ #
def test_a_set_made_from_a_preset_or_library_entry_records_where_it_came_from(tmp_path):
    from detail.api import skeletons as library

    proj = Experiment.create("p", tmp_path)
    from_preset = proj.create_example_set("side", "quadruped_side_5")
    assert from_preset.skeleton == PRESETS["quadruped_side_5"][1]
    assert from_preset.skeleton_ref == library.ref("quadruped_side_5")
    mine = library.create_skeleton("two", SK)
    from_library = proj.create_example_set("two", mine["id"])
    assert from_library.skeleton_ref == {
        "id": mine["id"],
        "name": "two",
        "digest": SK.digest(),
    }
    assert proj.example_set(from_library.id).skeleton_ref == from_library.skeleton_ref
    assert proj.create_example_set("loose", SK).skeleton_ref is None, "names no entry"
    copied = proj.create_example_set("again", from_library.id)
    assert copied.skeleton_ref == from_library.skeleton_ref, "another set's ref follows"
    assert (
        proj.example_sets()[0].id == from_preset.id
    ), "a preset is copied, never listed"


def test_resolve_skeleton_takes_a_preset_a_set_a_file_or_a_skeleton(tmp_path):
    proj = Experiment.create("p", tmp_path)
    made = proj.create_example_set("rats", SK)
    path = tmp_path / "sk.yaml"
    path.write_text(yaml.dump(SK.to_dict()), encoding="utf-8")

    assert kp.resolve_skeleton(proj, SK) == (SK, None)
    preset, ref = kp.resolve_skeleton(proj, "quadruped_overhead_12")
    assert (
        preset == PRESETS["quadruped_overhead_12"][1]
        and ref["id"] == "quadruped_overhead_12"
    )
    assert kp.resolve_skeleton(proj, made.id) == (
        SK,
        None,
    ), "another set's copy and ref"
    assert kp.resolve_skeleton(proj, path) == (SK, None)
    assert kp.resolve_skeleton(proj, SK.to_dict()) == (SK, None)
    with pytest.raises(
        ValueError,
        match="no skeleton preset, library skeleton, example set or file 'mice'",
    ):
        kp.resolve_skeleton(proj, "mice")
    (tmp_path / "empty.yaml").write_text("name: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="holds no skeleton"):
        kp.resolve_skeleton(proj, tmp_path / "empty.yaml")


# --- a set's skeleton is fixed; its name is not -------------------------------------- #
def _marked(proj):
    """A set with one frame carrying marks on both points."""
    from detail.core.examples import ExampleFrame

    made = proj.create_example_set("rats", SK)
    current = example_store.read_set(proj.dir, made.id)
    current.frames.append(
        ExampleFrame(
            image="frames/x.png",
            recording_id="recording_001",
            tracking_run_id="track_01",
            track_id=0,
            part_id="part_001",
            frame_idx=3,
            t_ns=3000,
            bbox=(0.1, 0.1, 0.5, 0.5),
            marks=[
                ExampleMark(target="nose", x=0.2, y=0.3),
                ExampleMark(target="tail_base", visible=False),
            ],
        )
    )
    example_store.write_set(proj.dir, current)
    return made.id


def test_update_renames_the_set_and_keeps_its_marks(tmp_path):
    proj = Experiment.create("p", tmp_path)
    set_id = _marked(proj)
    updated = proj.update_example_set(set_id, name="overhead rats")
    assert updated.name == "overhead rats" and updated.skeleton == SK
    assert proj.example_set(set_id).frames[0].mark_for("nose").x == 0.2, "marks kept"
    with pytest.raises(ValueError, match="needs a name"):
        proj.update_example_set(set_id, name="  ")
    with pytest.raises(ValueError, match="nothing to change"):
        proj.update_example_set(set_id)


def test_the_skeleton_changes_until_the_first_mark(tmp_path):
    """Marks name points, so a set's skeleton is fixed from its first mark; before that the
    marking window may still change it (the user's review, option C)."""
    from detail.core.examples import ExampleFrame

    proj = Experiment.create("p", tmp_path)
    made = proj.create_example_set("rats", "quadruped_side_5")
    changed = proj.update_example_set(made.id, skeleton="midline_3")
    assert changed.skeleton == PRESETS["midline_3"][1]
    assert changed.skeleton_ref["id"] == "midline_3", "the ref follows the new skeleton"
    assert proj.example_set(made.id).skeleton_ref["id"] == "midline_3", "written"

    current = example_store.read_set(proj.dir, made.id)
    current.frames.append(
        ExampleFrame(
            image="frames/x.png",
            recording_id="recording_001",
            tracking_run_id="track_01",
            track_id=0,
            part_id="part_001",
            frame_idx=3,
            t_ns=3000,
            bbox=(0.1, 0.1, 0.5, 0.5),
        )
    )
    example_store.write_set(proj.dir, current)
    assert (
        proj.update_example_set(made.id, skeleton="quadruped_side_5").skeleton_ref["id"]
        == "quadruped_side_5"
    ), "a frame without marks holds no points, so it does not fix the skeleton"

    set_id = _marked(proj)
    with pytest.raises(ValueError, match="marks on 1 frame.*make a new example set"):
        proj.update_example_set(set_id, skeleton="midline_3")
    assert (
        proj.update_example_set(set_id, skeleton=SK).skeleton == SK
    ), "unchanged is fine"


def test_duplicate_copies_the_skeleton_and_no_frames(tmp_path):
    proj = Experiment.create("p", tmp_path)
    set_id = _marked(proj)
    copy = proj.duplicate_example_set(set_id)
    assert copy.id != set_id and copy.name == "rats copy"
    assert copy.skeleton_ref is None, "a set made from a bare Skeleton names no entry"
    linked = proj.create_example_set("linked", "quadruped_side_5")
    assert proj.duplicate_example_set(linked.id).skeleton_ref == linked.skeleton_ref
    assert copy.skeleton == SK and copy.frames == []
    assert proj.duplicate_example_set(set_id, name="side").name == "side"


def test_export_then_import_round_trips_a_set_with_its_frames(tracked, tmp_path):
    proj = tracked.experiment
    made = proj.create_example_set("rats", SK)
    proj.add_example_frame(made.id, "recording_002", "track_01", "part_001", 2, 0)
    proj.set_example_marks(made.id, 0, [ExampleMark(target="nose", x=0.25, y=0.5)])

    written = proj.export_example_set(made.id, tmp_path / "out")

    assert written == tmp_path / "out" / "examples.yaml"
    assert written.read_text(encoding="utf-8-sig").startswith("schema_version: 1\n")
    with pytest.raises(FileExistsError, match="choose an empty folder"):
        proj.export_example_set(made.id, tmp_path / "out")
    back = proj.import_example_set(written)
    assert back.id != made.id and back.skeleton == SK
    assert len(back.frames) == 1 and back.frames[0].mark_for("nose").x == 0.25


def test_a_bare_skeleton_file_imports_as_an_empty_set(tmp_path):
    proj = Experiment.create("p", tmp_path)
    path = tmp_path / "side_view.yaml"
    path.write_text(
        yaml.dump(PRESETS["quadruped_side_5"][1].to_dict()), encoding="utf-8"
    )
    made = proj.import_example_set(path)
    assert made.name == "side_view" and made.frames == []
    assert made.skeleton == PRESETS["quadruped_side_5"][1]
    assert made.skeleton.layout == PRESETS["quadruped_side_5"][1].layout


# --- example sets ------------------------------------------------------------------- #
def test_create_example_set_copies_the_named_skeleton(tmp_path):
    proj = Experiment.create("p", tmp_path)

    created = proj.create_example_set("overhead", "quadruped_overhead_12")

    assert created.id == "ex_01"
    assert created.skeleton == PRESETS["quadruped_overhead_12"][1]
    assert [s.id for s in proj.example_sets()] == ["ex_01"]
    assert proj.example_set("ex_01").name == "overhead"
    audit = proj.create_example_set("au", SK, purpose="audit")
    assert [s.id for s in proj.example_sets()] == ["ex_01"]
    assert [s.id for s in proj.example_sets("audit")] == [audit.id]
    assert len(proj.example_sets(None)) == 2


def test_sampling_resolves_the_only_tracking_run_and_persists_the_frames(tracked):
    proj = tracked.experiment
    created = proj.create_example_set("rats", SK)
    stages = []

    updated = proj.sample_example_frames(
        created.id,
        ["recording_002"],
        3,
        strategy="spread",
        progress=lambda stage, fraction, detail: stages.append(stage),
    )

    assert len(updated.frames) == 3
    assert {f.tracking_run_id for f in updated.frames} == {"track_01"}
    assert len(proj.example_set(created.id).frames) == 3
    assert stages and stages[0] == "sampling frames"
    assert kp.resolve_sources(proj, {"recording_002": None}) == {
        "recording_002": "track_01"
    }
    assert kp.resolve_sources(proj, {"recording_002": "track_09"}) == {
        "recording_002": "track_09"
    }


def test_sampling_refuses_to_guess_between_tracking_runs(tracked):
    proj = tracked.experiment
    created = proj.create_example_set("rats", SK)
    sdir = tracked.tracked_recording.dir
    data = recording_store.read_recording(sdir)
    data["tracking_runs"].append(
        {"id": "track_02", "status": "complete", "detection_run_id": "det_01"}
    )
    recording_store.write_recording(sdir, data)

    with pytest.raises(ValueError, match="2 tracking runs .*say which one"):
        proj.sample_example_frames(created.id, ["recording_002"], 1, strategy="spread")
    data["tracking_runs"] = []
    recording_store.write_recording(sdir, data)
    with pytest.raises(ValueError, match="no complete tracking run"):
        proj.sample_example_frames(created.id, ["recording_002"], 1, strategy="spread")
    with pytest.raises(KeyError):
        proj.sample_example_frames(created.id, ["nowhere"], 1, strategy="spread")


def test_sampling_refuses_an_audit_set(tracked):
    proj = tracked.experiment
    audit = proj.create_example_set("au", SK, purpose="audit")
    with pytest.raises(ValueError, match="audit set"):
        proj.sample_example_frames(audit.id, ["recording_002"], 1, strategy="spread")


def test_frames_are_added_marked_and_removed_through_the_experiment(tracked):
    proj = tracked.experiment
    created = proj.create_example_set("rats", SK)

    updated = proj.add_example_frame(
        created.id, "recording_002", "track_01", "part_001", 2, 0
    )
    assert [f.frame_idx for f in updated.frames] == [2]
    assert updated.frames[0].sampled_by is None

    marked = proj.set_example_marks(
        created.id,
        0,
        [
            {"target": "nose", "x": 0.5, "y": 0.5},
            ExampleMark(target="tail_base", visible=False),
        ],
    )
    assert marked.labeled(0)
    assert proj.example_set(created.id).frames[0].mark_for("tail_base").visible is False

    with pytest.raises(ValueError, match="no keypoint 'ear'"):
        proj.set_example_marks(created.id, 0, [{"target": "ear", "x": 0.1, "y": 0.1}])
    with pytest.raises(ValueError, match="marked twice"):
        proj.set_example_marks(
            created.id,
            0,
            [
                {"target": "nose", "x": 0.1, "y": 0.1},
                {"target": "nose", "x": 0.2, "y": 0.2},
            ],
        )
    with pytest.raises(ValueError, match="off the frame"):
        proj.set_example_marks(created.id, 0, [{"target": "nose", "x": 1.5, "y": 0.1}])
    with pytest.raises(ValueError, match="has no x, y"):
        proj.set_example_marks(created.id, 0, [{"target": "nose"}])
    with pytest.raises(IndexError, match="no frame 3"):
        proj.set_example_marks(created.id, 3, [])
    assert proj.example_set(created.id).frames[0].mark_for("nose").x == 0.5

    image = proj.example_frame_image(created.id, 0)
    assert image.shape == (40, 40, 3)
    png = updated.frames[0].image
    emptied = proj.remove_example_frame(created.id, 0)
    assert emptied.frames == []
    assert (example_store.set_dir(proj.dir, created.id) / png).is_file(), "frames stay"
    with pytest.raises(IndexError):
        proj.remove_example_frame(created.id, 0)


def test_import_copies_a_prepared_set_in_and_keeps_its_digest(tracked, tmp_path):
    proj = tracked.experiment
    created = proj.create_example_set("rats", SK)
    proj.add_example_frame(created.id, "recording_002", "track_01", "part_001", 1, 0)
    proj.set_example_marks(
        created.id,
        0,
        [
            {"target": "nose", "x": 0.3, "y": 0.4},
            {"target": "tail_base", "visible": False},
        ],
    )
    source_dir = example_store.set_dir(proj.dir, created.id)
    export = tmp_path / "shared"
    export.mkdir()
    (export / "frames").mkdir()
    body = yaml.safe_load(
        (source_dir / "examples.yaml").read_text(encoding="utf-8-sig")
    )
    image = body["frames"][0]["image"]
    (export / image).write_bytes((source_dir / image).read_bytes())
    (export / "set.yaml").write_text(yaml.dump(body, sort_keys=False), encoding="utf-8")

    imported = proj.import_example_set(export / "set.yaml")

    assert imported.id == "ex_02"
    assert imported.name == "rats"
    assert len(imported.frames) == 1 and imported.frames[0].mark_for("nose").x == 0.3
    stored = yaml.safe_load(
        (example_store.set_dir(proj.dir, imported.id) / "examples.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    assert (
        stored["digest"] == body["digest"]
    ), "same images, marks and keys: same identity"


def test_import_refuses_a_missing_image_stale_marks_and_the_wrong_schema(tmp_path):
    proj = Experiment.create("p", tmp_path)
    body = {
        "schema_version": 1,
        "name": "x",
        "skeleton": SK.to_dict(),
        "frames": [
            {
                "image": "frames/missing.png",
                "recording_id": "recording_001",
                "tracking_run_id": "track_01",
                "track_id": 0,
                "part_id": "part_001",
                "frame_idx": 0,
                "t_ns": 0,
                "bbox": [0.1, 0.1, 0.5, 0.5],
                "marks": [],
            }
        ],
    }
    path = tmp_path / "set.yaml"
    path.write_text(yaml.dump(body), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="frames/missing.png"):
        proj.import_example_set(path)

    (tmp_path / "frames").mkdir()
    import cv2

    cv2.imwrite(str(tmp_path / "frames" / "missing.png"), np.zeros((8, 8, 3), np.uint8))
    body["frames"][0]["marks"] = [{"target": "ear", "x": 0.1, "y": 0.1}]
    path.write_text(yaml.dump(body), encoding="utf-8")
    with pytest.raises(ValueError, match="marks for \\['ear'\\]"):
        proj.import_example_set(path)

    body["schema_version"] = 2
    path.write_text(yaml.dump(body), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        proj.import_example_set(path)
    with pytest.raises(FileNotFoundError):
        proj.import_example_set(tmp_path / "nope.yaml")
    assert proj.example_sets() == []


def test_delete_through_the_experiment(tmp_path):
    proj = Experiment.create("p", tmp_path)
    created = proj.create_example_set("rats", SK)
    proj.delete_example_set(created.id)
    assert proj.example_sets() == []
    with pytest.raises(FileNotFoundError):
        proj.example_set(created.id)


# --- the stages' API wrappers (§26 *Surfaces*) ------------------------------------- #
def test_run_keypoint_detect_shapes_the_config_for_the_module(tracked, monkeypatch):
    """Keypoint detect runs on a **detection** run (§26, revised 2026-10-21): the wrapper
    names it and nothing about tracks."""
    from detail.core.registries import KEYPOINT_MODELS
    from tests.pipeline.test_keypoint_detect import _FakeKeypointModel

    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp", _FakeKeypointModel)
    recording = tracked.tracked_recording
    seen = {}
    monkeypatch.setattr(
        type(recording),
        "run",
        lambda self, module_id, config: seen.update(module_id=module_id, config=config),
    )
    recording.run_keypoint_detect(
        "det_01", "fake_kp", every_n_frames=2, batch_size=4, min_pose_confidence=0.3
    )
    assert seen["module_id"] == "keypoint_detect"
    config = seen["config"]
    assert config["detection_run_id"] == "det_01" and config["model_id"] == "fake_kp"
    assert "tracking_run_id" not in config and "track_ids" not in config
    assert config["every_n_frames"] == 2 and config["batch_size"] == 4
    assert config["min_pose_confidence"] == 0.3 and config["example_set_id"] is None
    assert config["compute"].backend == "cpu"  # the fake's only backend


def test_run_keypoint_track_names_the_tracking_run_and_the_match(tracked, monkeypatch):
    """A track run assigns a detect run's poses to a tracking run's subjects: the wrapper
    carries the tracking run (None: the stage resolves the one) and the match settings
    only when given, so the config model's defaults stay the defaults."""
    recording = tracked.tracked_recording
    seen = {}
    monkeypatch.setattr(
        type(recording),
        "run",
        lambda self, module_id, config: seen.update(module_id=module_id, config=config),
    )
    recording.run_keypoint_track(
        "kd_01",
        tracking="track_01",
        method="despike_smooth",
        match_min_inside=0.6,
        match_mask_grow=0.1,
    )
    assert seen["module_id"] == "keypoint_track"
    assert seen["config"] == {
        "detect_run_id": "kd_01",
        "tracking_run_id": "track_01",
        "method": "despike_smooth",
        "config": {},
        "match_min_inside": 0.6,
        "match_mask_grow": 0.1,
    }
    recording.run_keypoint_track("kd_01")
    assert seen["config"] == {
        "detect_run_id": "kd_01",
        "tracking_run_id": None,
        "method": None,
        "config": {},
    }


def test_keypoint_instances_reads_a_parts_unassigned_instances(tracked, monkeypatch):
    """``keypoint_instances(kd, part)`` replaces ``keypoint_candidates(kd, track)``: every
    instance of a part with its box, detection row and candidates, and no subject."""
    from detail.core.registries import KEYPOINT_MODELS
    from tests.pipeline.test_keypoint_detect import _FakeKeypointModel, _config, _run

    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp", _FakeKeypointModel)
    recording = tracked.tracked_recording
    _run(tracked, recording, _config())
    rows = recording.keypoint_instances("kd_01", "part_001")
    assert rows["frame_idx"].tolist() == list(range(N_FRAMES))
    assert rows["detection_row"].tolist() == list(range(N_FRAMES))
    assert "track_id" not in rows and "object_id" not in rows
    assert rows["candidates"].shape == (N_FRAMES, 2, 2, 2)
    assert rows["bbox"].shape == (N_FRAMES, 4) and "extras/mask_area" in rows
    with pytest.raises(KeyError, match="no video part 'part_009'"):
        recording.keypoint_instances("kd_01", "part_009")
    import shutil

    shutil.rmtree(recording.dir / "keypoint_detect" / "kd_01" / "part_001.zarr")
    empty = recording.keypoint_instances("kd_01", "part_001")
    assert empty["frame_idx"].shape == (0,) and empty["points"].shape == (0, 2, 2)
    assert empty["candidates"].shape == (0, 2, 2, 2), "a part nothing was placed in"


def test_an_imported_set_keeps_its_skeleton_link_only_on_a_digest_match(tmp_path):
    """A skeleton id means something only on the machine that minted it. An imported set
    keeps ``skeleton_ref`` only when a library entry here holds the set's skeleton, and then
    names that entry; otherwise it carries none and is listed by its digest."""
    from detail.api import skeletons as library
    from detail.storage import skeleton_library

    here = library.create_skeleton("rats", SK)
    other = library.create_skeleton("fly", Skeleton(names=("a", "b", "c")))
    a = Experiment.create("a", tmp_path / "a")
    b = Experiment.create("b", tmp_path / "b")
    made = a.create_example_set("rats", here["id"])
    body = yaml.safe_load(
        (example_store.set_dir(a.dir, made.id) / "examples.yaml").read_text(
            encoding="utf-8-sig"
        )
    )

    def imported(ref):
        path = tmp_path / f"set_{len(list(tmp_path.glob('set_*')))}.yaml"
        path.write_text(yaml.dump({**body, "skeleton_ref": ref}), encoding="utf-8")
        return b.import_example_set(path).skeleton_ref

    assert imported(body["skeleton_ref"]) == library.ref(here["id"])
    # minted elsewhere under an id that names a different skeleton here
    assert imported({**body["skeleton_ref"], "id": other["id"]}) == library.ref(
        here["id"]
    )
    # a ref whose digest is not the set's skeleton
    assert imported({**body["skeleton_ref"], "digest": "0" * 16}) is None
    skeleton_library.remove(here["id"])  # "another machine": nothing here holds it
    assert imported(body["skeleton_ref"]) is None
