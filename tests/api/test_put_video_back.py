"""Putting a copied video back into an experiment that was imported without it (#29).

**What was missing.** A slim bundle (the default export) carries each copied part's entry and
timestamp map but not its file. On the machine that imported it, `recording verify` reported
``missing_file`` and nothing could fix that: `Recording.relink_video_part` refused a copied part
("nothing to relink"), the GUI's Locate video listed referenced parts only, and Replace deleted
the copied video with no way back.

**The owner's decision, which these tests pin.** Locate video and ``recording relink`` accept a
copied part whose file is missing and copy the chosen file back in; the part stays copied. A file
is accepted only if its frame times match the part's stored timestamp map exactly (count and every
``t_ns``, both re-based to 0) and its resolution matches. Newly copied parts store a
``fingerprint`` (the referenced part's shape, `integrity.file_fingerprint`), and a part that has
one must match it too. Locate offers the file the part was copied from first. A cancel leaves no
half file.

The surfaces are tested where they live: the CLI in `tests/test_cli.py`, the GUI dialog and page
in `tests/gui/test_pages.py`, the Export dialog's note in `tests/gui/test_export.py`.
"""

import cv2
import numpy as np
import pytest

from detail.api.experiment import Experiment
from detail.core.errors import UserInputError
from detail.core.progress import OperationCancelled
from detail.storage import export as export_store
from detail.storage import integrity
from detail.storage import locations
from detail.storage import recording as recording_store
from detail.storage import video_integrity
from detail.storage import video_roots

H, W = 24, 32


def _write_video(path, n_frames, *, fps=30.0, size=(W, H), shade=17):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert vw.isOpened()
    for i in range(n_frames):
        vw.write(np.full((size[1], size[0], 3), (i * shade) % 256, dtype=np.uint8))
    vw.release()
    return path


def _isolate_storage(tmp_path, monkeypatch, name="cfg"):
    monkeypatch.setattr(locations, "CONFIG_DIR", tmp_path / name)
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / name / "storage.yaml")


def _parts(recording):
    return recording_store.read_recording(recording.dir)["video"]["files"]


def _slim_copy(tmp_path, monkeypatch, n_parts=1):
    """``study`` with copied video, exported slim and imported as ``study_copy``: the state a
    user meets on the second machine. Returns ``(the imported recording, its source files)``.
    """
    _isolate_storage(tmp_path, monkeypatch)
    experiment = Experiment.create("study", tmp_path / "a")
    recording = experiment.create_recording("recording_001")
    sources = []
    for i in range(n_parts):
        src = _write_video(tmp_path / "raw" / f"m{i}.mp4", 6 + 3 * i)
        recording.add_video_part(src)
        sources.append(src)
    archive = experiment.export(tmp_path / "study.tar")  # slim: no video
    placed = export_store.import_experiment(archive, tmp_path / "b", name="study_copy")
    copy = Experiment(placed).get_recording("recording_001")
    assert [i.kind for i in copy.verify_videos()] == ["missing_file"] * n_parts
    return copy, sources


# --- a copied part records its fingerprint ---------------------------------------------- #
def test_a_copied_part_records_the_fingerprint_of_its_copy(tmp_path, monkeypatch):
    _isolate_storage(tmp_path, monkeypatch)
    recording = Experiment.create("p", tmp_path).create_recording("recording_001")
    recording.add_video_part(_write_video(tmp_path / "raw" / "m.mp4", 5))

    (part,) = _parts(recording)

    copy = recording.dir / part["file"]
    assert integrity.same_content(integrity.file_fingerprint(copy), part["fingerprint"])
    assert (
        "source" not in part
    ), "a copied part must not look referenced to the resolver"
    assert recording.verify_videos() == []


# --- putting it back --------------------------------------------------------------------- #
def test_a_slim_imported_copy_is_put_back_from_its_original(tmp_path, monkeypatch):
    copy, (src,) = _slim_copy(tmp_path, monkeypatch)
    before = _parts(copy)[0]
    reported = []

    registered = copy.relink_video_part(
        0, src, progress=lambda *event: reported.append(event)
    )

    assert registered is None
    (part,) = _parts(copy)
    dest = copy.dir / "video" / "part_001.mp4"
    assert dest.read_bytes() == src.read_bytes()
    assert (
        part["file"] == before["file"] and "source" not in part
    ), "still a copied part"
    assert part["origin"] == before["origin"]
    assert integrity.same_content(integrity.file_fingerprint(dest), part["fingerprint"])
    assert copy.verify_videos() == [] and copy.video_available
    assert [stage for stage, *_ in reported][0] == "Reading timestamps"
    assert reported[-1][:2] == ("Copying", 1.0)
    assert sorted(p.name for p in dest.parent.iterdir()) == [
        "part_001.mp4",
        "timestamps",
    ]


def test_a_later_part_is_compared_on_its_own_clock(tmp_path, monkeypatch):
    """The stored map of part 2 starts where part 1 ended; the file's starts at 0. Compared
    without re-basing, the right file for any part after the first would be refused."""
    copy, sources = _slim_copy(tmp_path, monkeypatch, n_parts=2)
    stored = recording_store.TimestampMap.from_csv(
        copy.dir / _parts(copy)[1]["timestamps"]["generated"]
    ).values
    assert stored[0] > 0, "the fixture must give part 2 an offset to re-base"

    copy.relink_video_part(1, sources[1])

    assert (copy.dir / "video" / "part_002.mp4").read_bytes() == sources[1].read_bytes()
    assert [i.part_id for i in copy.verify_videos()] == ["part_001"]


@pytest.mark.parametrize(
    "make, says",
    [
        (
            lambda p: _write_video(p, 7),
            "it has 7 frames, the recorded part has 6",
        ),
        (
            lambda p: _write_video(p, 6, fps=25.0),
            "its frame times differ from the recorded part's from frame 1 (40.000 ms into "
            "the part, recorded 33.333 ms)",
        ),
        (
            lambda p: _write_video(p, 6, size=(2 * W, 2 * H)),
            f"it is {2 * W}×{2 * H}, the recording is {W}×{H}",
        ),
        (
            lambda p: (p.write_bytes(b"not a video"), p)[1],
            "it cannot be read as a video",
        ),
    ],
    ids=["frame count", "frame times", "resolution", "not a video"],
)
def test_a_file_that_is_not_the_parts_video_is_refused_naming_what_differs(
    tmp_path, monkeypatch, make, says
):
    copy, _ = _slim_copy(tmp_path, monkeypatch)
    # Taken off the part so the frame-time check is the one under test (an older copied part
    # has no fingerprint; the fingerprint check is tested on its own below).
    data = recording_store.read_recording(copy.dir)
    del data["video"]["files"][0]["fingerprint"]
    recording_store.write_recording(copy.dir, data)
    before = (copy.dir / "recording.yaml").read_bytes()
    (tmp_path / "other").mkdir()
    other = make(tmp_path / "other" / "x.mp4")

    with pytest.raises(UserInputError) as refused:
        copy.relink_video_part(0, other)

    assert str(refused.value).startswith(
        f"{other} is not the video of part 0 (part_001): {says}"
    )
    assert not (copy.dir / "video" / "part_001.mp4").exists()
    assert (copy.dir / "recording.yaml").read_bytes() == before


def test_a_part_with_a_fingerprint_needs_the_same_content_as_well(
    tmp_path, monkeypatch
):
    """Same frame times, different pictures: a part that recorded its fingerprint refuses it.
    One that predates fingerprints has only the frame times to go on, and accepts it."""
    copy, _ = _slim_copy(tmp_path, monkeypatch)
    lookalike = _write_video(tmp_path / "other" / "same_timing.mp4", 6, shade=41)

    with pytest.raises(UserInputError, match="content differs from the file that was"):
        copy.relink_video_part(0, lookalike)

    data = recording_store.read_recording(copy.dir)
    del data["video"]["files"][0]["fingerprint"]
    recording_store.write_recording(copy.dir, data)
    copy.relink_video_part(0, lookalike)  # the frame times match: only content differed
    assert integrity.same_content(
        integrity.file_fingerprint(lookalike), _parts(copy)[0]["fingerprint"]
    ), "a part put back records the fingerprint of what it now holds"


def test_a_copied_part_whose_video_is_here_is_refused(tmp_path, monkeypatch):
    _isolate_storage(tmp_path, monkeypatch)
    recording = Experiment.create("p", tmp_path).create_recording("recording_001")
    src = _write_video(tmp_path / "raw" / "m.mp4", 5)
    recording.add_video_part(src)

    with pytest.raises(UserInputError, match="its video is already here"):
        recording.relink_video_part(0, src)


def test_a_cancelled_put_back_leaves_nothing(tmp_path, monkeypatch):
    copy, (src,) = _slim_copy(tmp_path, monkeypatch)
    before = (copy.dir / "recording.yaml").read_bytes()

    def cancel_while_copying(stage, _fraction, _detail):
        if stage == "Copying":
            raise OperationCancelled()

    with pytest.raises(OperationCancelled):
        copy.relink_video_part(0, src, progress=cancel_while_copying)

    assert sorted(p.name for p in (copy.dir / "video").iterdir()) == ["timestamps"]
    assert (copy.dir / "recording.yaml").read_bytes() == before


# --- what Locate is offered -------------------------------------------------------------- #
def test_parts_to_locate_offers_where_a_missing_copy_came_from(tmp_path, monkeypatch):
    copy, (src,) = _slim_copy(tmp_path, monkeypatch)

    (part,) = copy.parts_to_locate()

    assert (part.index, part.stem, part.copied) == (0, "part_001", True)
    assert part.copied_from == src.resolve()
    assert part.origin_name == "m0.mp4"


def test_parts_to_locate_offers_nothing_this_machine_cannot_find(tmp_path, monkeypatch):
    """Another computer: the origin's root is not registered here, so nothing is offered, and
    the file's name is still there for the dialog to ask for."""
    copy, _ = _slim_copy(tmp_path, monkeypatch)
    _isolate_storage(tmp_path, monkeypatch, name="cfg_elsewhere")
    assert video_roots.list_video_roots() == {}

    (part,) = copy.parts_to_locate()

    assert part.copied and part.copied_from is None and part.origin_name == "m0.mp4"


def test_parts_to_locate_lists_referenced_parts_and_only_missing_copies(
    tmp_path, monkeypatch
):
    _isolate_storage(tmp_path, monkeypatch)
    recording = Experiment.create("p", tmp_path).create_recording("recording_001")
    recording.add_video_part(_write_video(tmp_path / "raw" / "a.mp4", 5))
    recording.add_video_part(
        _write_video(tmp_path / "raw" / "b.mp4", 5), storage="reference"
    )

    assert [(p.index, p.copied) for p in recording.parts_to_locate()] == [(1, False)]
    (recording.dir / "video" / "part_001.mp4").unlink()
    assert [(p.index, p.copied) for p in recording.parts_to_locate()] == [
        (0, True),
        (1, False),
    ]


# --- the frame-time rule itself ---------------------------------------------------------- #
def test_part_video_mismatch_compares_maps_on_their_own_clock():
    local = np.array([0, 33, 67, 100])
    assert (
        video_integrity.part_video_mismatch(
            local + 5_000, local, recorded_resolution=[4, 3], resolution=[4, 3]
        )
        is None
    )
    assert video_integrity.part_video_mismatch(
        local, np.array([0, 33, 68, 100]), recorded_resolution=None, resolution=[4, 3]
    ).startswith("its frame times differ from the recorded part's from frame 2 ")
    assert (
        video_integrity.part_video_mismatch(
            [], [], recorded_resolution=[4, 3], resolution=[4, 3]
        )
        is None
    )


# --- an export asked for video that packs none ------------------------------------------- #
def test_video_not_packed_says_why(tmp_path, monkeypatch):
    copy, (src,) = _slim_copy(tmp_path, monkeypatch)
    experiment = Experiment(copy.dir.parent.parent)

    assert experiment.video_not_packed() == (
        "1 recording(s) have no video on this machine (put it back with Locate video, "
        "or `detail recording relink`)"
    )
    archive = experiment.export(tmp_path / "again.tar", include_video=True)
    contents = export_store.read_manifest(archive)["contents"]
    assert (contents["video_requested"], contents["video_included"]) == (True, False)

    copy.relink_video_part(0, src)

    assert experiment.video_not_packed() is None


def test_video_not_packed_names_video_used_in_place(tmp_path, monkeypatch):
    _isolate_storage(tmp_path, monkeypatch)
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.create_recording("recording_001")
    recording.add_video_part(
        _write_video(tmp_path / "raw" / "m.mp4", 5), storage="reference"
    )

    assert experiment.video_not_packed() == (
        "1 recording(s) use their video in place, and video used in place never travels"
    )
