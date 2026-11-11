"""A recording's id is the tool's; its name is the user's (§11).

The id used to be the video filename's stem, so the *file* was the identity: nowhere to
put a better label, an arbitrary choice for a recording spanning several files, and no way
to correct a bad one. These pin the split, and the forward-compatibility of every
experiment created before it.
"""

import numpy as np
import pytest

from detail.api.experiment import Experiment
from detail.storage import experiment as experiment_store


def _video(path, frames=4):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (16, 16))
    for _ in range(frames):
        writer.write(np.zeros((16, 16, 3), np.uint8))
    writer.release()
    return path


# --------------------------------------------------------------------------- #
# Minting ids
# --------------------------------------------------------------------------- #
def test_ids_are_assigned_in_order(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    ids = [experiment.add_recording(f"clip_{i}").id for i in range(3)]
    assert ids == ["recording_000", "recording_001", "recording_002"]


def test_a_deleted_recordings_id_is_never_handed_out_again(tmp_path):
    """The counter is persisted rather than derived from the highest existing id. Reuse
    would point every run record, export and analysis row naming it at a different
    animal."""
    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("first")
    second = experiment.add_recording("second")
    experiment.delete_recording(second.id)

    assert experiment.add_recording("third").id == "recording_002"


def test_minting_skips_an_id_that_already_exists(tmp_path):
    """A hand-made recording_000 (or one restored from a bundle) can't be collided with."""
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_000")  # chosen explicitly, counter untouched
    assert experiment.add_recording("mine").id == "recording_001"


def test_create_recording_still_takes_the_id_it_is_given(tmp_path):
    """The library contract scripts and tests rely on: the caller chooses the key."""
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.create_recording("r1")
    assert recording.id == "r1" and experiment.get_recording("r1").name == "r1"


# --------------------------------------------------------------------------- #
# Unique names, two ways
# --------------------------------------------------------------------------- #
def test_the_tool_suffixes_a_name_it_chose(tmp_path):
    """Two files of the same name in different folders is routine; failing the import
    over a label would not be."""
    experiment = Experiment.create("p", tmp_path)
    names = [experiment.add_recording("m27_day3").name for _ in range(3)]
    assert names == ["m27_day3", "m27_day3_2", "m27_day3_3"]


def test_a_name_a_user_typed_is_refused_not_renamed(tmp_path):
    """Silently changing what someone deliberately typed is the worse failure."""
    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("m27_day3")
    other = experiment.add_recording("m27_day4")

    with pytest.raises(ValueError, match="already named"):
        other.set_name("m27_day3")
    assert experiment.get_recording(other.id).name == "m27_day4"  # unchanged


def test_renaming_a_recording_to_its_own_name_is_not_a_collision(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("m27_day3")
    recording.set_name("m27_day3")  # no-op, must not raise


# --------------------------------------------------------------------------- #
# Resolving what a user types
# --------------------------------------------------------------------------- #
def test_resolve_takes_a_name_or_an_id(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("m27_day3")
    assert experiment.resolve_recording("m27_day3").id == recording.id
    assert experiment.resolve_recording(recording.id).name == "m27_day3"


def test_an_id_beats_a_name_that_impersonates_it(tmp_path):
    """A recording perversely *named* another's id must still be reachable by its own."""
    experiment = Experiment.create("p", tmp_path)
    first = experiment.add_recording("first")
    impostor = experiment.add_recording(first.id)  # named "recording_000"

    assert experiment.resolve_recording(first.id).id == first.id
    assert impostor.id != first.id


def test_resolving_something_that_does_not_exist_says_what_does(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("m27_day3")
    with pytest.raises(KeyError, match="m27_day3"):
        experiment.resolve_recording("ghost")


# --------------------------------------------------------------------------- #
# Experiments made before any of this
# --------------------------------------------------------------------------- #
def test_an_experiment_with_filename_derived_ids_keeps_working(tmp_path):
    """Forward-only: an id is an opaque string, so nothing migrates. Its recordings
    resolve, list, carry metadata, and new ones mint alongside them."""
    experiment = Experiment.create("p", tmp_path)
    legacy = experiment.create_recording("m27_day3", metadata={"dose": "0.5"})
    assert "next_recording_index" not in experiment_store.read_config(experiment.dir)

    fresh = experiment.add_recording("m27_day4")

    # Sorted by id, so a legacy filename-derived one can sort either side of a minted one.
    assert sorted(r.id for r in experiment.recordings) == [legacy.id, fresh.id]
    assert experiment.resolve_recording("m27_day3").id == "m27_day3"
    assert experiment.recording_metadata["m27_day3"]["dose"] == "0.5"
    assert fresh.id == "recording_000"


# --------------------------------------------------------------------------- #
# Where the footage came from
# --------------------------------------------------------------------------- #
def test_a_copied_part_records_where_it_came_from(tmp_path):
    """Provenance, not a location: the tool holds the footage, so nothing reads this to
    find pixels — but "which file was this?" still needs an answer once the name has been
    edited away from the filename."""
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("m27_day3")
    source = _video(tmp_path / "m27_day3.mp4")
    recording.add_video_part(source)

    assert recording.video_sources == [str(source.resolve())]


def test_a_copied_parts_origin_is_stored_relative_and_shown_absolute(tmp_path):
    """Tenet 6: the experiment file carries {root, rel} through the machine-local video-root
    registry, and the absolute path is composed at display time — the same mechanism a
    referenced part uses, so an experiment stays portable."""
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("m27_day3")
    recording.add_video_part(_video(tmp_path / "m27_day3.mp4"))

    text = (recording.dir / "recording.yaml").read_text(encoding="utf-8-sig")
    assert str(tmp_path) not in text  # no absolute path in the experiment file
    part = recording.video_parts[0]
    assert part["origin"]["rel"] == "m27_day3.mp4"
    # ...and not under `source`, which is what resolve_part_video_path reads to find the
    # pixels — a copy's reads must stay on the copy, or deleting the original would break
    # runs we hold the footage for.
    assert "source" not in part
    assert recording.video_parts[0]["file"] == "video/part_001.mp4"


def test_a_recording_with_no_video_has_no_source(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    assert experiment.add_recording("planned").video_sources == []
