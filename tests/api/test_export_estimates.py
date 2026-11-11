"""Every export target says how big it will be and how long it will take, before it starts.

**The class: a cost the user cannot see until it has been paid.** An export that wrote one
video per recording of a whole run did not fit the drive it was written to (issue #710). Nothing
said so beforehand: not the dialog, not `detail export run`. The fix is not a warning on that one target. It is a question
every target has to answer: ``ExportTarget.estimate(subject, scope, options)``, asked through
``exporters.estimate``, shown by the Export dialog and printed by ``export run``, and compared
with the free space on the destination drive.

What these pin:

- **every shipped target declares an estimator** (a registry sweep, with the negative case, so a
  twenty-second target cannot ship without one);
- an estimate is an estimate: it follows the scope (twice the recordings is more), it matches what
  the export really writes where it can be counted, a video target times real frames on this
  machine, and a target that cannot know says *why* instead of showing a number;
- **one target's trouble is reported beside it**: an estimator that raises is a "could not
  estimate", and the others still answer;
- it writes nothing: no export folder, and the disk-speed probe's file is gone when it returns;
- the free-space check refuses an export that will not fit, on both surfaces that start one.

How accurate: checked against real exports. Size and time come out close, except a video's
time, which reads high (a timed sample includes warm-up the long run does not), and the
figure-drawing targets, which are looser. The constants and where they came
from are in `detail/api/export_estimates.py`. These tests do not pin the *time* of any target:
a timing assertion on the machine that also runs the suite is a flake by construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from detail.api import export_estimates, exporters
from detail.api.exporters import (
    Estimate,
    ExportEstimate,
    ExportScope,
    ExportSubject,
    ExportTarget,
)


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
def test_every_shipped_target_declares_an_estimator():
    missing = [
        f"{t.subject_kind}/{t.id}"
        for t in exporters.all_targets()
        if t.estimate is None
    ]
    assert not missing, f"targets with no estimator: {missing}"


def test_the_sweep_would_catch_a_target_with_none(monkeypatch):
    bare = ExportTarget(
        id="bare",
        label="Bare",
        subject_kind="run",
        artifact="x",
        artifact_is_data=True,
        run=lambda *a, **k: [],
    )
    monkeypatch.setattr(exporters, "_TARGETS", [bare])
    assert [t.id for t in exporters.all_targets() if t.estimate is None] == ["bare"]


def test_a_target_with_no_estimator_answers_with_the_reason_not_a_number():
    bare = ExportTarget(
        id="bare",
        label="Bare",
        subject_kind="run",
        artifact="x",
        artifact_is_data=True,
        run=lambda *a, **k: [],
    )
    result = exporters.estimate(ExportSubject("run", "/nowhere", "r"), bare)
    ((_, only),) = result.per_target
    assert only.bytes is None and only.seconds is None
    assert "no estimate" in only.basis


def test_an_estimator_that_raises_is_reported_beside_the_others():
    def boom(subject, scope, options):
        raise RuntimeError("the video is gone")

    ok = ExportTarget(
        id="ok",
        label="Ok",
        subject_kind="run",
        artifact="x",
        artifact_is_data=True,
        run=lambda *a, **k: [],
        estimate=lambda s, sc, o: Estimate(10, 1.0, "fine"),
    )
    bad = ExportTarget(
        id="bad",
        label="Bad",
        subject_kind="run",
        artifact="x",
        artifact_is_data=True,
        run=lambda *a, **k: [],
        estimate=boom,
    )
    result = exporters.estimate(ExportSubject("run", "/nowhere", "r"), [ok, bad])
    by_id = dict(result.per_target)
    assert by_id["ok"] == Estimate(10, 1.0, "fine")
    assert by_id["bad"].bytes is None
    assert (
        "could not estimate" in by_id["bad"].basis
        and "the video is gone" in by_id["bad"].basis
    )
    assert result.bytes == 10 and not result.complete  # a floor, and it says so


# --------------------------------------------------------------------------- #
# the totals, and how they are said
# --------------------------------------------------------------------------- #
def test_totals_and_whether_it_fits():
    result = ExportEstimate(
        (("a", Estimate(100, 2.0)), ("b", Estimate(50, 3.0))), free_bytes=150
    )
    assert (result.bytes, result.seconds, result.complete) == (150, 5.0, True)
    assert result.fits is True
    assert ExportEstimate(result.per_target, free_bytes=149).fits is False
    assert ExportEstimate(result.per_target, free_bytes=None).fits is None
    assert ExportEstimate((("a", Estimate(None, None)),), free_bytes=10).fits is None


def test_sizes_are_said_one_way():
    assert exporters.format_size(None) == "unknown"
    assert exporters.format_size(512) == "512 B"
    assert exporters.format_size(1_500) == "1.5 KB"
    assert exporters.format_size(3_700_000_000) == "3.7 GB"
    assert exporters.format_size(117_500_000_000) == "117.5 GB"


def test_a_summary_says_about_for_a_whole_and_at_least_for_a_floor():
    whole = ExportEstimate((("a", Estimate(3_700_000_000, 252.0)),))
    assert exporters.describe_estimate(whole) == "about 3.7 GB, about 4m12s"
    floor = ExportEstimate(
        (("a", Estimate(3_700_000_000, 252.0)), ("b", Estimate(None, None, "no video")))
    )
    assert exporters.describe_estimate(floor).startswith("at least 3.7 GB, at least ")
    nothing = ExportEstimate((("b", Estimate(None, None, "no video")),))
    assert exporters.describe_estimate(nothing) == "cannot be estimated"
    assert exporters.describe_one(Estimate(None, None, "no video")) == (
        "cannot be estimated: no video"
    )


def test_free_space_of_a_folder_that_does_not_exist_yet_is_its_drives(monkeypatch):
    """Asked of the nearest folder that exists. Asserted on *which folder is measured*, not by
    comparing two live readings: free space moves between two calls while the rest of the suite
    writes files, which failed this test twice in one day by 8 KB."""
    import shutil

    measured = []
    real = shutil.disk_usage

    def recording(path):
        measured.append(Path(path))
        return real(path)

    monkeypatch.setattr(shutil, "disk_usage", recording)
    nowhere = "this-folder-does-not-exist-yet/nor-this"
    assert exporters.free_bytes_at(nowhere) > 0
    assert measured == [Path(".")]


# --------------------------------------------------------------------------- #
# following the scope, and matching what is written
# --------------------------------------------------------------------------- #
@pytest.fixture
def two(processed_recording):
    built = processed_recording
    square = built.square(4, 20, 4, 20)
    built.add_recording(
        "recording_002",
        detections=[(i, 0, (0.1, 0.1, 0.5, 0.5), square) for i in range(4)],
        object_ids=[0, 0, 0, 0],
    )
    return built


def _estimate(built, run_id, target_id, scope=None, **options):
    subject = ExportSubject("run", built.experiment.dir, run_id)
    result = exporters.estimate(
        subject, exporters.target("run", target_id), scope=scope, **options
    )
    ((_, only),) = result.per_target
    return only


@pytest.mark.parametrize(
    "run_attribute, target_id",
    [("detection_run_id", "masks"), ("tracking_run_id", "tracks")],
)
def test_an_estimate_follows_the_recordings_asked_for(two, run_attribute, target_id):
    run_id = getattr(two, run_attribute)
    everything = _estimate(two, run_id, target_id)
    big = _estimate(
        two, run_id, target_id, ExportScope(recording_ids=("recording_002",))
    )
    small = _estimate(
        two, run_id, target_id, ExportScope(recording_ids=("recording_001",))
    )

    # recording_002 has four detections to recording_001's one
    assert small.bytes < big.bytes < everything.bytes
    assert everything.bytes == pytest.approx(small.bytes + big.bytes, rel=0.05)
    assert everything.seconds >= big.seconds >= small.seconds > 0


def test_a_scope_naming_nothing_in_this_run_estimates_nothing(two):
    nothing = _estimate(
        two,
        two.detection_run_id,
        "masks",
        ExportScope(recording_ids=("recording_999",)),
    )
    assert nothing.bytes == 0


def test_masks_size_is_close_to_what_is_written(two, tmp_path):
    """Counted rows times a sampled row size: within a factor of two even on one-row fixtures,
    where the fixed JSON around a mask is most of the file."""
    scope = ExportScope(recording_ids=("recording_002",))
    estimated = _estimate(two, two.detection_run_id, "masks", scope).bytes
    subject = ExportSubject("run", two.experiment.dir, two.detection_run_id)
    result = exporters.write_export(
        subject, [exporters.target("run", "masks")], tmp_path, scope=scope
    )
    actual = (result.dest / "recording_002_masks.json").stat().st_size
    assert 0.5 <= estimated / actual <= 2.0, (estimated, actual)


def test_vectors_size_is_counted_not_guessed(two, make_embedding_run, tmp_path):
    import numpy as np
    from detail.storage import runs as runs_store

    ids = ("recording_001", "recording_002")
    for rid in ids:
        make_embedding_run(
            two.experiment.get_recording(rid).dir,
            np.ones((5, 8), dtype=np.float32),
            np.arange(5),
            np.arange(5) * 1000,
            embed_run_id="emb_07",
        )
    runs_store.write_run(
        two.experiment.dir,
        "emb_07",
        {
            "id": "emb_07",
            "module_id": "embedding",
            "recording_ids": list(ids),
            "per_recording": {
                rid: {"run_id": "emb_07", "status": "complete"} for rid in ids
            },
            "status": "complete",
        },
    )
    subject = ExportSubject("run", two.experiment.dir, "emb_07")
    estimated = exporters.estimate(subject, exporters.target("run", "vectors")).bytes
    result = exporters.write_export(
        subject, [exporters.target("run", "vectors")], tmp_path
    )
    actual = sum(
        (result.dest / name).stat().st_size
        for name in ("embeddings.npy", "embeddings_index.csv")
    )
    assert 0.8 <= estimated / actual <= 1.3, (estimated, actual)


def test_the_bundle_estimate_is_what_it_packs(processed_recording, tmp_path):
    subject = ExportSubject("experiment", processed_recording.experiment.dir)
    target = exporters.target("experiment", "bundle")
    estimated = exporters.estimate(subject, target).bytes
    result = exporters.write_export(subject, [target], tmp_path)
    (tar,) = [f for f in result.files if f.endswith(".tar")]
    actual = (result.dest / tar).stat().st_size
    assert 0.8 <= estimated / actual <= 1.5, (estimated, actual)


def test_a_gzipped_bundle_is_timed_at_gzips_speed(
    processed_recording, tmp_path, monkeypatch
):
    """gzip packs at ~58 MB/s against a disk's hundreds, so ticking it must move the time, and
    the basis must say why. Same bytes either way: the estimate is of what is packed."""
    from detail.api import export_estimates

    monkeypatch.setattr(
        export_estimates, "disk_rate", lambda _d: (1.0e9, "a test disk at 1000 MB/s")
    )
    subject = ExportSubject("experiment", processed_recording.experiment.dir)
    target = exporters.target("experiment", "bundle")
    plain = exporters.estimate(subject, target).per_target[0][1]
    packed = exporters.estimate(subject, target, gzip=True).per_target[0][1]
    assert packed.bytes == plain.bytes
    assert packed.seconds > plain.seconds
    assert "gzip" in packed.basis


# --------------------------------------------------------------------------- #
# the keypoint formats: counted rows, a sample of the real writer, and the two that cannot be sized
# --------------------------------------------------------------------------- #
from tests.api.test_export_recordings_scope import (  # noqa: E402,F401  (fixtures)
    keypoint_runs,
    two_keypoint_recordings,
)


def _kp_estimate(it, target_id, scope=None, **options):
    subject = ExportSubject("run", it.experiment.dir, "run_011")
    result = exporters.estimate(
        subject, exporters.target("run", target_id), scope=scope, **options
    )
    ((_, only),) = result.per_target
    return only


def test_the_keypoint_long_table_is_counted_and_sized_from_the_real_writer(
    two_keypoint_recordings, tmp_path
):
    it = two_keypoint_recordings
    both = _kp_estimate(it, "keypoints")
    one = _kp_estimate(it, "keypoints", ExportScope(recording_ids=("recording_003",)))

    assert "28 long-table rows" in both.basis and "14 long-table rows" in one.basis
    assert one.bytes < both.bytes
    subject = ExportSubject("run", it.experiment.dir, "run_011")
    result = exporters.write_export(
        subject, [exporters.target("run", "keypoints")], tmp_path
    )
    actual = sum(
        (result.dest / f).stat().st_size for f in result.files if f.endswith(".csv")
    )
    assert 0.5 <= both.bytes / actual <= 2.0, (both.bytes, actual)


def test_a_deeplabcut_table_is_counted_in_cells_and_follows_the_scope(
    two_keypoint_recordings, tmp_path
):
    it = two_keypoint_recordings
    both = _kp_estimate(it, "keypoints_dlc")
    one = _kp_estimate(
        it, "keypoints_dlc", ExportScope(recording_ids=("recording_003",))
    )

    # 5 frames x 2 tracks x 2 points x (x, y, likelihood) = 60 cells a recording
    assert "120 cells" in both.basis and "60 cells" in one.basis
    assert one.bytes < both.bytes
    subject = ExportSubject("run", it.experiment.dir, "run_011")
    result = exporters.write_export(
        subject, [exporters.target("run", "keypoints_dlc")], tmp_path
    )
    actual = sum(
        (result.dest / f).stat().st_size for f in result.files if f.endswith("_DLC.csv")
    )
    assert 0.4 <= both.bytes / actual <= 2.5, (both.bytes, actual)


@pytest.mark.parametrize("target_id", ["keypoints_sleap", "keypoints_nwb"])
def test_a_compressed_keypoint_format_says_it_cannot_be_sized_and_blocks_nothing(
    two_keypoint_recordings, target_id
):
    """A figure several times too big would switch the dialog's Export off for an export that
    fits. These two say they cannot know, and the total becomes a floor."""
    it = two_keypoint_recordings
    estimate = _kp_estimate(it, target_id)

    assert estimate.bytes is None and estimate.seconds is None
    assert "compresses its arrays" in estimate.basis
    total = ExportEstimate((("a", Estimate(10, 1.0)), (target_id, estimate)), 100)
    assert not total.complete, "the total is a floor"
    assert total.fits is True, "the unknown adds nothing to what is refused on"


def test_a_keypoint_video_is_timed_on_real_frames_plus_its_csv(
    two_keypoint_recordings,
):
    it = two_keypoint_recordings
    both = _kp_estimate(it, "keypoints_video")
    one = _kp_estimate(
        it, "keypoints_video", ExportScope(recording_ids=("recording_003",))
    )
    overlay = _kp_estimate(it, "keypoints_video", overlay=True)

    assert both.bytes and both.seconds and "timed on" in both.basis
    assert "plus keypoints.csv" in both.basis
    assert both.bytes > one.bytes > 0
    assert overlay.seconds > both.seconds, "the tracking overlay costs more per frame"


# --------------------------------------------------------------------------- #
# video: timed on this machine
# --------------------------------------------------------------------------- #
def _write_video(built, recording_id="recording_001", frames=30):
    """A real 64x64 mp4 where the recording's first video part says it is (the fixture
    registers the part and leaves the file out, which is what a missing video is)."""
    import cv2
    import numpy as np

    path = built.experiment.get_recording(recording_id).dir / "video" / "part_001.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64))
    noise = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(noise.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def test_a_video_estimate_is_timed_on_real_frames_and_scales_with_the_frames_asked_for(
    processed_recording,
):
    _write_video(processed_recording)
    subject = ExportSubject(
        "recording", processed_recording.experiment.dir, "recording_001"
    )
    target = exporters.target("recording", "clip")

    long = exporters.estimate(
        subject, target, scope=ExportScope(frames=(0, 999))
    ).per_target[0][1]
    short = exporters.estimate(
        subject, target, scope=ExportScope(frames=(0, 99))
    ).per_target[0][1]

    assert long.bytes and long.seconds and long.seconds > 0
    assert "timed on" in long.basis and "1,000 frames" in long.basis
    assert long.bytes > short.bytes > 0  # same sample, ten times the frames


def test_a_recording_whose_video_is_not_here_cannot_be_sized_and_says_so(
    processed_recording,
):
    """The fixture registers a video part and writes no file: the state of a recording whose
    video lives on a drive that is not attached."""
    subject = ExportSubject(
        "recording", processed_recording.experiment.dir, "recording_001"
    )
    target = exporters.target("recording", "clip")

    sized = exporters.estimate(
        subject, target, scope=ExportScope(frames=(0, 9))
    ).per_target[0][1]

    assert sized.bytes is None and sized.seconds is None
    assert "recording_001" in sized.basis and "not on this machine" in sized.basis


def test_a_clip_with_no_frame_range_estimates_nothing_to_write(processed_recording):
    _write_video(processed_recording)
    subject = ExportSubject(
        "recording", processed_recording.experiment.dir, "recording_001"
    )
    target = exporters.target("recording", "clip")

    none = exporters.estimate(subject, target).per_target[0][1]
    some = exporters.estimate(
        subject, target, scope=ExportScope(frames=(0, 3)), embed_run_id="x"
    ).per_target[0][1]

    assert none.bytes == 0 and "no frame range" in none.basis
    assert some.bytes and some.bytes > 0


# --------------------------------------------------------------------------- #
# it writes nothing
# --------------------------------------------------------------------------- #
def test_the_disk_probe_leaves_nothing_behind_and_a_bad_path_gets_a_default(tmp_path):
    rate, basis = export_estimates.disk_rate(tmp_path)
    assert rate > 0 and "MB/s" in basis
    assert list(tmp_path.iterdir()) == [], "the probe file is deleted"

    default_rate, default_basis = export_estimates.disk_rate(None)
    assert default_rate == export_estimates.DEFAULT_DISK_BYTES_PER_S
    assert "assumes" in default_basis


def test_estimating_creates_no_export_folder(two, tmp_path):
    out = tmp_path / "exports"
    exporters.estimate(
        ExportSubject("run", two.experiment.dir, two.detection_run_id),
        exporters.target("run", "masks"),
        parent=out,
    )
    assert not out.exists()


def test_estimate_carries_the_free_space_of_the_destination(two, tmp_path):
    result = exporters.estimate(
        ExportSubject("run", two.experiment.dir, two.detection_run_id),
        exporters.target("run", "masks"),
        parent=tmp_path / "later",
    )
    # the drive is live (this test's own files, other processes): the same drive, not the same byte
    assert result.free_bytes == pytest.approx(
        exporters.free_bytes_at(tmp_path), abs=64 * 1024 * 1024
    )
    assert result.fits is True


def test_the_estimates_say_where_every_figure_came_from():
    """A number nobody can check is one they stop trusting the first time it is wrong."""
    source = Path(export_estimates.__file__).read_text(encoding="utf-8-sig")
    for constant in (
        "MASKS_ROWS_PER_S",
        "TRACKS_ROWS_PER_S",
        "VECTORS_ROWS_PER_S",
        "INSTANCES_BASE_BYTES",
    ):
        index = source.index(constant)
        assert (
            "#:" in source[max(0, index - 700) : index]
        ), f"{constant} has no measurement note"


# --------------------------------------------------------------------------- #
# the command line
# --------------------------------------------------------------------------- #
def _cli(two, tmp_path, *args):
    """``export run`` on the two-recording detection run (``recording_001``, ``recording_002``)."""
    from tests.test_cli_export import _run

    return _run(
        "export", "run", "--experiment", str(two.experiment.dir),
        "--run", two.detection_run_id,
        "--out", str(tmp_path / "out"), *args,
    )  # fmt: skip


def _written(tmp_path):
    return sorted(p.name for p in (tmp_path / "out").glob("*/*_masks.json"))


def test_the_cli_prints_an_estimate_and_writes_nothing_when_only_asked(two, tmp_path):
    result = _cli(two, tmp_path, "--target", "masks", "--estimate")

    assert result.exit_code == 0, result.output
    assert "estimate: about" in result.output
    assert "masks:" in result.output  # a line per target, with how it was reached
    assert "free on the drive" in result.output
    assert not (
        tmp_path / "out"
    ).exists(), "--estimate writes nothing, not even the folder"


def test_the_cli_narrows_by_recording_and_the_estimate_follows(two, tmp_path):
    both = _cli(two, tmp_path, "--target", "masks", "--estimate")
    one = _cli(
        two,
        tmp_path,
        "--target",
        "masks",
        "--for-recording",
        "recording_002",
        "--estimate",
    )

    def total(output):
        return next(l for l in output.splitlines() if l.startswith("estimate:"))

    assert both.exit_code == one.exit_code == 0
    assert total(both.output) != total(one.output)


def test_the_cli_writes_only_the_recordings_asked_for(two, tmp_path):
    result = _cli(
        two, tmp_path, "--target", "masks", "--for-recording", "recording_002"
    )
    assert result.exit_code == 0, result.output
    written = _written(tmp_path)
    assert written == ["recording_002_masks.json"]


def test_a_damaged_frame_view_is_left_out_of_the_estimate_not_fatal(
    authored_cohort, monkeypatch
):
    """The estimate is advice; one view the export itself will report on must not take the
    figure for every other recording down with it (`storage.dirscan`'s rule for listings).
    """
    from detail.storage import frame_view

    subject = ExportSubject("analysis_run", authored_cohort.experiment.dir, "an_01")
    target = exporters.target("analysis_run", "frame_labels")
    whole = exporters.estimate(subject, target).per_target[0][1]

    real = frame_view.read_frame_view

    def damaged_for_r1(run_dir, recording_id, unit_subject):
        if recording_id == "r1":
            raise ValueError("zero-filled")
        return real(run_dir, recording_id, unit_subject)

    monkeypatch.setattr(frame_view, "read_frame_view", damaged_for_r1)
    partial = exporters.estimate(subject, target).per_target[0][1]

    assert partial.bytes < whole.bytes
    assert "1 unreadable view(s) left out" in partial.basis


def test_the_cli_takes_several_recordings(two, tmp_path):
    result = _cli(
        two,
        tmp_path,
        "--target", "masks",
        "--for-recording", "recording_001",
        "--for-recording", "recording_002",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    written = _written(tmp_path)
    assert written == ["recording_001_masks.json", "recording_002_masks.json"]


def test_the_cli_refuses_a_recording_the_run_does_not_have(two, tmp_path):
    result = _cli(two, tmp_path, "--target", "masks", "--for-recording", "nope")
    assert result.exit_code != 0
    assert "nope" in result.output


def test_the_cli_refuses_an_export_that_will_not_fit_unless_told_otherwise(
    two, tmp_path, monkeypatch
):
    monkeypatch.setattr(exporters, "free_bytes_at", lambda path: 10)

    refused = _cli(two, tmp_path, "--target", "masks")
    assert refused.exit_code != 0
    assert "is free where" in refused.output and "--ignore-free-space" in refused.output
    assert not (tmp_path / "out").exists()

    allowed = _cli(two, tmp_path, "--target", "masks", "--ignore-free-space")
    assert allowed.exit_code == 0, allowed.output
    assert _written(tmp_path)


def test_estimate_only_reports_a_tight_drive_without_refusing(
    two, tmp_path, monkeypatch
):
    monkeypatch.setattr(exporters, "free_bytes_at", lambda path: 10)
    result = _cli(two, tmp_path, "--target", "masks", "--estimate")
    assert result.exit_code == 0, result.output
    assert "free on the drive" in result.output
