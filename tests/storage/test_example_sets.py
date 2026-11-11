"""Example-set storage (§26): copied-in frames named by content, an atomically rewritten YAML,
and a delete guard that reads the run records rather than a marker kept up to date."""

import os

import numpy as np
import pytest
import yaml

from detail.core.examples import ExampleFrame, ExampleMark
from detail.core.skeleton import Skeleton
from detail.storage import example_sets as store
from detail.storage import experiment as experiment_store

SK = Skeleton(
    names=("nose", "left_ear", "right_ear"), pairs=(("left_ear", "right_ear"),)
)


def _rgb(seed: int, h: int = 24, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _frame(image: str, **over) -> ExampleFrame:
    fields = dict(
        image=image,
        recording_id="recording_001",
        tracking_run_id="track_01",
        track_id=0,
        part_id="part_001",
        frame_idx=3,
        t_ns=3000,
        bbox=(0.1, 0.1, 0.5, 0.5),
    )
    fields.update(over)
    return ExampleFrame(**fields)


def _yaml(experiment_dir, set_id) -> str:
    return (store.set_dir(experiment_dir, set_id) / store.EXAMPLES_FILENAME).read_text(
        encoding="utf-8-sig"
    )


# --- creating ----------------------------------------------------------------- #
def test_create_writes_schema_version_first(tmp_path):
    created = store.create_set(tmp_path, "rats", SK)

    text = _yaml(tmp_path, created.id)
    assert text.startswith("schema_version: 1")
    data = yaml.safe_load(text)
    assert list(data)[:2] == ["schema_version", "id"]
    assert data["purpose"] == "example"
    assert data["frames"] == []


def test_ids_count_up_and_live_under_examples(tmp_path):
    a = store.create_set(tmp_path, "a", SK)
    b = store.create_set(tmp_path, "b", SK)

    assert (a.id, b.id) == ("ex_01", "ex_02")
    assert store.set_dir(tmp_path, a.id).parent == experiment_store.examples_dir(
        tmp_path
    )
    assert store.discover_sets(tmp_path) == ["ex_01", "ex_02"]


def test_create_holds_a_copy_of_the_skeleton(tmp_path):
    created = store.create_set(tmp_path, "rats", SK)
    assert created.skeleton == SK
    assert created.skeleton is not SK
    assert store.read_set(tmp_path, created.id).skeleton.digest() == SK.digest()


def test_create_refuses_an_empty_name_a_bad_purpose_and_a_bad_skeleton(tmp_path):
    with pytest.raises(ValueError, match="needs a name"):
        store.create_set(tmp_path, "  ", SK)
    with pytest.raises(ValueError, match="purpose"):
        store.create_set(tmp_path, "x", SK, purpose="training")
    with pytest.raises(ValueError, match="appears twice"):
        store.create_set(tmp_path, "x", Skeleton(names=("a", "a")))
    assert store.discover_sets(tmp_path) == []


def test_an_audit_set_is_listed_by_its_purpose(tmp_path):
    store.create_set(tmp_path, "ex", SK)
    audit = store.create_set(tmp_path, "au", SK, purpose="audit")

    assert [s.id for s in store.list_sets(tmp_path, purpose="audit")] == [audit.id]
    assert len(store.list_sets(tmp_path)) == 2


# --- frame images --------------------------------------------------------------- #
def test_frame_images_are_named_by_content_and_written_once(tmp_path):
    created = store.create_set(tmp_path, "rats", SK)

    first = store.add_frame_image(tmp_path, created.id, _rgb(1))
    again = store.add_frame_image(tmp_path, created.id, _rgb(1))
    other = store.add_frame_image(tmp_path, created.id, _rgb(2))

    assert first == again
    assert first != other
    assert first.startswith("frames/") and first.endswith(".png")
    frames = store.set_dir(tmp_path, created.id) / store.FRAMES_DIR
    assert sorted(p.name for p in frames.iterdir()) == sorted(
        [first.split("/")[1], other.split("/")[1]]
    )
    sha = store._file_sha256(store.set_dir(tmp_path, created.id) / first)
    assert first == f"frames/{sha[:16]}.png"


def test_load_image_round_trips_pixels_losslessly(tmp_path):
    created = store.create_set(tmp_path, "rats", SK)
    rgb = _rgb(3)
    image = store.add_frame_image(tmp_path, created.id, rgb)

    np.testing.assert_array_equal(store.load_image(tmp_path, created.id, image), rgb)


def test_load_image_missing_names_the_file(tmp_path):
    created = store.create_set(tmp_path, "rats", SK)
    with pytest.raises(FileNotFoundError, match="frames/nope.png"):
        store.load_image(tmp_path, created.id, "frames/nope.png")


# --- the digest ----------------------------------------------------------------- #
def test_digest_follows_marks_and_images_not_the_name(tmp_path):
    created = store.create_set(tmp_path, "rats", SK)
    image = store.add_frame_image(tmp_path, created.id, _rgb(1))
    created.frames.append(_frame(image))
    store.write_set(tmp_path, created)
    before = yaml.safe_load(_yaml(tmp_path, created.id))["digest"]

    created.name = "renamed"
    store.write_set(tmp_path, created)
    assert yaml.safe_load(_yaml(tmp_path, created.id))["digest"] == before

    created.frames[0].marks.append(ExampleMark(target="nose", x=0.5, y=0.5))
    store.write_set(tmp_path, created)
    after = yaml.safe_load(_yaml(tmp_path, created.id))["digest"]
    assert after != before


def test_digest_follows_an_image_swapped_under_its_name(tmp_path):
    created = store.create_set(tmp_path, "rats", SK)
    image = store.add_frame_image(tmp_path, created.id, _rgb(1))
    created.frames.append(_frame(image))
    store.write_set(tmp_path, created)
    before = yaml.safe_load(_yaml(tmp_path, created.id))["digest"]

    path = store.set_dir(tmp_path, created.id) / image
    path.write_bytes(b"not the same picture")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    store.write_set(tmp_path, created)

    assert yaml.safe_load(_yaml(tmp_path, created.id))["digest"] != before


# --- reading ------------------------------------------------------------------- #
def test_unknown_fields_survive_a_read_and_a_write(tmp_path):
    created = store.create_set(tmp_path, "rats", SK)
    image = store.add_frame_image(tmp_path, created.id, _rgb(1))
    created.frames.append(
        _frame(image, marks=[ExampleMark(target="nose", x=0.1, y=0.2)])
    )
    store.write_set(tmp_path, created)
    path = store.set_dir(tmp_path, created.id) / store.EXAMPLES_FILENAME
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    data["later_set_field"] = {"a": 1}
    data["frames"][0]["later_frame_field"] = "kept"
    data["frames"][0]["marks"][0]["later_mark_field"] = 7
    path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")

    reread = store.read_set(tmp_path, created.id)
    store.write_set(tmp_path, reread)

    written = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    assert written["later_set_field"] == {"a": 1}
    assert written["frames"][0]["later_frame_field"] == "kept"
    assert written["frames"][0]["marks"][0]["later_mark_field"] == 7


def test_read_unknown_set_names_the_existing_ones(tmp_path):
    store.create_set(tmp_path, "rats", SK)
    with pytest.raises(FileNotFoundError, match=r"ex_01"):
        store.read_set(tmp_path, "ex_09")


def test_a_torn_file_costs_only_its_row(tmp_path):
    good = store.create_set(tmp_path, "rats", SK)
    torn = store.create_set(tmp_path, "torn", SK)
    (store.set_dir(tmp_path, torn.id) / store.EXAMPLES_FILENAME).write_text(
        "schema_version: 1\nid: ex_02\nframes: [", encoding="utf-8"
    )

    assert [s.id for s in store.list_sets(tmp_path)] == [good.id]
    with pytest.raises(yaml.YAMLError):
        store.read_set(tmp_path, torn.id)


# --- deleting --------------------------------------------------------------------- #
def _list_in_a_run(
    experiment_dir, set_id, recording_id="recording_001", run_id="kd_01"
):
    run_dir = (
        experiment_store.recording_path(experiment_dir, recording_id)
        / "keypoint_detect"
        / run_id
    )
    run_dir.mkdir(parents=True)
    (run_dir / "detect.yaml").write_text(
        yaml.dump(
            {"schema_version": 1, "id": run_id, "example_set": {"id": set_id}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_delete_is_refused_while_a_run_lists_the_set(tmp_path):
    created = store.create_set(tmp_path, "rats", SK)
    _list_in_a_run(tmp_path, created.id)

    assert store.runs_using(tmp_path, created.id) == [("recording_001", "kd_01")]
    with pytest.raises(ValueError, match="kd_01 \\(recording_001\\)"):
        store.delete_set(tmp_path, created.id)
    assert store.discover_sets(tmp_path) == [created.id]


def test_delete_removes_a_set_no_run_lists(tmp_path):
    created = store.create_set(tmp_path, "rats", SK)
    other = store.create_set(tmp_path, "other", SK)
    _list_in_a_run(tmp_path, other.id)

    store.delete_set(tmp_path, created.id)

    assert store.discover_sets(tmp_path) == [other.id]
    with pytest.raises(FileNotFoundError):
        store.delete_set(tmp_path, created.id)
