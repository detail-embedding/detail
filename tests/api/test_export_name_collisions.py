"""Two things in one export never share a file name (#29, findings 9 and 10).

**The class: a write whose target is not known to be free.** An output name was built from a
lossy transform of text a user typed — a recording's name, a cluster's name — and nothing checked
that it was unique inside the one folder it was written to. ``Mouse 1 (pre)`` and
``Mouse 1 [pre]`` are two recordings and one stem (``Mouse_1_pre_``), so the second recording's
keypoint CSV, DLC table, SLEAP file, NWB file and video were written over the first's. The export
reported ``complete``, listed one set of files, and one recording was simply gone. Two subjects
matched on the same span of one recording did the same to behavior clips, one step worse: a
subject with no detection there wrote nothing, and its cleanup deleted the *other* subject's clip
by name.

**How it hid.** Every test exported one recording, or one subject, or names that were already
clean. Nothing is wrong at the line that builds the name; it is wrong only beside the other line
that builds the same one. A case-only difference (``Mouse`` / ``mouse``) is the same collision on
the filesystems most users have (macOS, Windows), and invisible on the Linux CI runs on.

**The rule.** A file name built from user text inside one output goes through
``core.utils.unique_stems``: names that clean to one stem (ignoring case) each carry their own key
— the recording id, the subject. And ``write_export`` refuses a target whose own file list names
one file twice, so a target added later that collides fails loudly instead of losing data.

**The guard.** Every registered export target is classified here: either it names files from
user text, and a test below runs it with two names that clean to one stem, or it names them from
ids or fixed names, with the reason that cannot collide. A new target fails until someone has
decided which it is.
"""

from __future__ import annotations

import csv
import importlib.util
import shutil

import pytest

from detail.api import behavior_clips, exporters
from detail.api.exporters import ExportSubject, ExportTarget
from detail.core.utils import new_dir, unique_stems
from detail.storage import behavior_output as bo
from detail.storage import recording as recording_store
from detail.storage import runs as runs_store
from tests.api.test_behavior_clips import _FakePreview, _instance
from tests.api.test_clip_render import make_part, write_indexed_video
from tests.api.test_keypoint_export import (  # noqa: F401 — a fixture, used by name
    DETECT_RUN,
    TRACK_RUN,
    keypoint_runs,
)

# --------------------------------------------------------------------------- #
# the registry: every target is classified
# --------------------------------------------------------------------------- #
#: Targets whose file names carry text a user typed, each run below with two names that clean to
#: one stem.
USER_NAMED = {
    ("run", "keypoints"),
    ("run", "keypoints_dlc"),
    ("run", "keypoints_sleap"),
    ("run", "keypoints_nwb"),
    ("run", "keypoints_video"),
    ("recording", "streams"),
    ("analysis_run", "clips"),
}

#: Every other target, and why two of its files cannot share a name inside one export folder.
ID_OR_FIXED_NAMED = {
    ("experiment", "bundle"): "one file, in a folder minted for this export",
    ("experiment", "metadata"): "one fixed name",
    ("recording", "timestamps"): "the recording id and the part number",
    ("recording", "video"): "the recording id and the part number",
    ("recording", "clip"): "one clip, named by recording id and frame range",
    ("recording", "geometry"): "the recording id",
    ("analysis_run", "instances"): "fixed names",
    ("analysis_run", "ethogram"): "the recording id",
    ("analysis_run", "frame_labels"): "the recording id and the subject number",
    ("analysis_run", "performance"): "fixed names",
    ("comparison", "report"): "the comparison's own tree, one folder per export",
    ("run", "vectors"): "fixed names",
    ("run", "tracks"): "the recording id",
    ("run", "masks"): "the recording id",
    ("run", "performance"): "fixed names",
    ("distill", "performance"): "fixed names",
}


def _unclassified(targets) -> set[tuple[str, str]]:
    return (
        {(t.subject_kind, t.id) for t in targets} - USER_NAMED - set(ID_OR_FIXED_NAMED)
    )


def test_every_export_target_says_how_its_file_names_stay_unique():
    exporters._load()
    assert not _unclassified(exporters.all_targets()), (
        "classify each new export target: USER_NAMED (and a collision test below) when a file "
        "name carries text a user typed, ID_OR_FIXED_NAMED with the reason otherwise"
    )
    stale = (USER_NAMED | set(ID_OR_FIXED_NAMED)) - {
        (t.subject_kind, t.id) for t in exporters.all_targets()
    }
    assert not stale, f"no longer registered, so take them out: {sorted(stale)}"


def test_the_classification_would_catch_a_new_target():
    new = ExportTarget(
        id="per_cluster_pngs",
        label="x",
        subject_kind="analysis_run",
        artifact="x",
        run=lambda *a, **k: [],
        artifact_is_data=True,
    )
    assert _unclassified([new]) == {("analysis_run", "per_cluster_pngs")}


# --------------------------------------------------------------------------- #
# the helpers
# --------------------------------------------------------------------------- #
def test_unique_stems_keys_only_the_names_that_collide():
    stems = unique_stems(
        ["Mouse 1 (pre)", "Mouse 1 [pre]", "Rat 2"],
        ["recording_001", "recording_002", "recording_003"],
    )
    assert stems == [
        "Mouse_1_pre__recording_001",
        "Mouse_1_pre__recording_002",
        "Rat_2",
    ]


def test_unique_stems_treats_a_case_only_difference_as_a_collision():
    """``Mouse`` and ``mouse`` are one file on macOS and Windows."""
    assert unique_stems(["Mouse", "mouse"], ["a", "b"]) == ["Mouse_a", "mouse_b"]


def test_unique_stems_is_unique_even_when_a_key_lands_on_another_name():
    stems = unique_stems(["x", "x", "x_a"], ["a", "b", "c"])
    assert len({s.casefold() for s in stems}) == 3
    assert stems[2] == "x_a", "a name that was already unique keeps its plain stem"


def test_unique_stems_keeps_non_ascii_names_apart():
    stems = unique_stems(["老鼠甲", "老鼠乙"], ["recording_001", "recording_002"])
    assert len(set(stems)) == 2


def test_new_dir_never_returns_an_existing_folder(tmp_path):
    first = new_dir(tmp_path / "parent", "export")
    second = new_dir(tmp_path / "parent", "export")
    assert first.is_dir() and second.is_dir() and first != second
    assert second.name == "export_2"


# --------------------------------------------------------------------------- #
# write_export: a target that names one file twice fails
# --------------------------------------------------------------------------- #
def test_a_target_that_writes_one_file_twice_is_reported_failed(tmp_path, monkeypatch):
    """The safety net under the per-target fix: a collision is loud, never a quiet loss."""

    def run(subject, dest, scope, progress, **options):
        out = dest / "Mouse.csv"
        out.write_text("a", encoding="utf-8")
        (dest / "mouse.csv").write_text("b", encoding="utf-8")
        return [out, dest / "mouse.csv"]

    target = ExportTarget(
        id="twice",
        label="Twice",
        subject_kind="experiment",
        artifact="x",
        run=run,
        artifact_is_data=True,
    )
    # A target outside the registry has no availability rule; this one is always available.
    monkeypatch.setattr(exporters, "unavailable_reason", lambda target, subject: None)
    experiment = tmp_path / "exp"
    experiment.mkdir()
    result = exporters.write_export(
        ExportSubject("experiment", experiment), target, tmp_path / "out"
    )
    (outcome,) = result.outcomes
    assert outcome.status == "failed"
    assert "mouse.csv" in outcome.error.lower() and "same file" in outcome.error


# --------------------------------------------------------------------------- #
# keypoints: two recordings whose names clean to one stem
# --------------------------------------------------------------------------- #
NAMES = {"recording_002": "Mouse 1 (pre)", "recording_003": "Mouse 1 [pre]"}

_EXTRA = all(importlib.util.find_spec(m) for m in ("h5py", "pynwb", "ndx_pose"))


@pytest.fixture
def two_colliding_recordings(keypoint_runs):  # noqa: F811
    """``keypoint_runs`` with its recording copied to a second, both run records naming both,
    and the two named so they clean to one file stem."""
    it, recording = keypoint_runs
    second = recording.dir.parent / "recording_003"
    shutil.copytree(recording.dir, second)
    for recording_id, name in NAMES.items():
        rdir = recording.dir.parent / recording_id
        data = recording_store.read_recording(rdir)
        data["id"], data["name"] = recording_id, name
        recording_store.write_recording(rdir, data)
    for run_id in (DETECT_RUN, TRACK_RUN):
        record = runs_store.read_run(it.experiment.dir, run_id)
        record["recording_ids"] = sorted(NAMES)
        stage = record["per_recording"]["recording_002"]
        record["per_recording"]["recording_003"] = dict(stage)
        runs_store.write_run(it.experiment.dir, run_id, record)
    return it


@pytest.mark.parametrize(
    "target_id",
    [
        "keypoints",
        "keypoints_dlc",
        pytest.param(
            "keypoints_sleap",
            marks=pytest.mark.skipif(not _EXTRA, reason="keypoints_export extra"),
        ),
        pytest.param(
            "keypoints_nwb",
            marks=pytest.mark.skipif(not _EXTRA, reason="keypoints_export extra"),
        ),
        "keypoints_video",
    ],
)
def test_two_recordings_whose_names_clean_alike_each_get_their_own_files(
    two_colliding_recordings, tmp_path, target_id
):
    it = two_colliding_recordings
    run_id = DETECT_RUN if target_id == "keypoints_video" else TRACK_RUN
    result = exporters.write_export(
        ExportSubject("run", it.experiment.dir, run_id),
        exporters.target("run", target_id),
        tmp_path / "out",
    )
    (outcome,) = result.outcomes
    assert outcome.status == "complete", outcome.error
    for recording_id in NAMES:
        mine = [f for f in outcome.files if recording_id in f]
        assert mine, f"{recording_id} has no file of its own: {outcome.files}"
        for name in mine:
            assert (result.dest / name).is_file()
    assert len({f.casefold() for f in outcome.files}) == len(outcome.files)


def test_the_long_table_of_each_recording_holds_only_that_recording(
    two_colliding_recordings, tmp_path
):
    it = two_colliding_recordings
    result = exporters.write_export(
        ExportSubject("run", it.experiment.dir, TRACK_RUN),
        exporters.target("run", "keypoints"),
        tmp_path / "out",
    )
    for recording_id in NAMES:
        (name,) = [f for f in result.files if recording_id in f and f.endswith(".csv")]
        with open(result.dest / name, newline="", encoding="utf-8-sig") as handle:
            ids = {row["recording_id"] for row in csv.DictReader(handle)}
        assert ids == {recording_id}


# --------------------------------------------------------------------------- #
# streams: names that differ only by case
# --------------------------------------------------------------------------- #
def test_two_streams_that_differ_only_by_case_get_two_files(tmp_path):
    """Stream names are already ``[A-Za-z0-9_-]``, so the only way two clean to one file name is
    case — one file on macOS and Windows. The store refuses that pair now; this checks the
    export's naming on its own, with the second stream planted past the refusal."""
    from detail.api import export_targets

    names = export_targets._stream_file_names("recording_001", ["EMG", "emg"])
    assert len({n.casefold() for n in names}) == 2


# --------------------------------------------------------------------------- #
# clips: two subjects on one span, and two cluster names that clean alike
# --------------------------------------------------------------------------- #
def _clip_export(tmp_path, rows, preview):
    dest = tmp_path / "out"
    dest.mkdir(exist_ok=True)
    written = behavior_clips.export_behavior_clips(
        tmp_path,
        dest,
        instances=rows,
        embed_run_ids={r.recording_id: "emb_01" for r in rows},
        preview=preview,
    )
    with open(dest / "clips.csv", encoding="utf-8-sig") as handle:
        table = list(csv.DictReader(handle))
    return dest, written, table


class _TwoSubjectPreview(_FakePreview):
    """Subject 0 and subject 99 both detected on every frame of the one part."""

    def __init__(self, video):
        super().__init__(video)
        frames = range(40)
        self._scrub._parts = [make_part(video, frames, other_frames=frames)]


@pytest.fixture
def clip_video(tmp_path):
    path = tmp_path / "part.mp4"
    write_indexed_video(path, 40)
    return path


def test_two_detected_subjects_on_one_span_get_two_clips(tmp_path, clip_video):
    rows = [
        _instance("r1", "q01", 2, 6, subject=0),
        _instance("r1", "q01", 2, 6, subject=99),
    ]
    dest, _written, table = _clip_export(tmp_path, rows, _TwoSubjectPreview(clip_video))
    files = [r["video_overlay"] for r in table]
    assert all(files) and len(set(files)) == 2, files
    assert all((dest / f).is_file() for f in files)


def test_an_undetected_subject_does_not_delete_the_other_subjects_clip(
    tmp_path, clip_video
):
    """Subject 1 has no detection on the span, so its clip is empty and is dropped — by its own
    name, not by the name subject 0's clip was written under."""
    rows = [
        _instance("r1", "q01", 2, 6, subject=0),
        _instance("r1", "q01", 2, 6, subject=1),
    ]
    dest, _written, table = _clip_export(tmp_path, rows, _FakePreview(clip_video))
    by_subject = {r["subject"]: r["video_overlay"] for r in table}
    assert by_subject["1"] == ""
    assert by_subject["0"] and (dest / by_subject["0"]).is_file()


def test_two_cluster_names_that_clean_alike_get_two_clips(tmp_path, clip_video):
    rows = [_instance("r1", "a/b", 2, 6), _instance("r1", "a_b", 2, 6)]
    dest, _written, table = _clip_export(tmp_path, rows, _FakePreview(clip_video))
    files = [r["video_overlay"] for r in table]
    assert all(files) and len(set(files)) == 2, files
    assert all((dest / f).is_file() for f in files)


def test_the_clip_name_still_leads_with_the_cluster():
    """Unchanged where nothing collides: a name-sorted folder still groups a cluster."""
    rows = [bo.BehaviorInstance("r1", 0, 2_000_000, 6_000_000, 2, 6, "c01", "")]
    assert behavior_clips.clip_names(rows) == ["c01_r1_000002-000006.mp4"]
