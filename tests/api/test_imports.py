"""Recording import orchestration — RecordingImportSpec + import_specs (Section 3)."""

import multiprocessing
from pathlib import Path

import cv2
import numpy as np
import pytest

from detail.api.imports import (
    RecordingImportSpec,
    StreamImportSpec,
    create_shell,
    import_specs,
)
from detail.api.experiment import Experiment

# Reuse the MED-PC sample shape from the stream tests.
DATA_TXT = (
    "Start Date: 02/16/26\nSubject: TEST1\nBox: 4\nStart Time: 14:00:38\n"
    "MSN: Test_MSN\nG:       3.000\nH:      10.000\nE:\n"
    "     0:       19.000       11.000      100.000\nT:\n"
    "     0:        0.000        1.500       10.000\n"
)
PROGRAM_MPC = "\\  Event stamps ( array E )\n\\   11 = a\n\\   19 = b\n\\   100 = end\n"


def _video(path, n_frames=6, w=32, h=24):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(n_frames):
        vw.write(np.full((h, w, 3), (i * 17) % 256, np.uint8))
    vw.release()
    return path


def test_metadata_only_spec_is_ready_immediately(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    spec = RecordingImportSpec(id="recording_001", metadata={"animal_id": "M001"})
    errors, cancelled = import_specs(experiment, [spec])
    assert errors == [] and cancelled is False
    recording = experiment.get_recording("recording_001")
    assert recording.import_status == "ready"
    assert recording.video_parts == []
    assert experiment.recording_metadata["recording_001"] == {
        "acquisition_setup": "",
        "animal_id": "M001",
    }


def test_create_shell_sets_queued_when_deferred_work(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    spec = RecordingImportSpec(
        id="recording_001", video_parts=[_video(tmp_path / "v.mp4")]
    )
    create_shell(experiment, spec)  # only the fast part — video import still pending
    assert experiment.get_recording("recording_001").import_status == "queued"


def test_create_shells_makes_whole_bout_visible(tmp_path):
    from detail.api.imports import create_shells

    experiment = Experiment.create("p", tmp_path)
    create_shells(
        experiment,
        [
            RecordingImportSpec(id="vid", video_parts=[_video(tmp_path / "v.mp4")]),
            RecordingImportSpec(id="empty", metadata={"animal_id": "M001"}),
        ],
    )
    assert {s.id for s in experiment.recordings} == {"vid", "empty"}
    assert experiment.get_recording("vid").import_status == "queued"
    assert experiment.get_recording("empty").import_status == "ready"


def test_import_pending_cancel_marks_remaining_cancelled_kept(tmp_path):
    from detail.api.imports import create_shells, import_pending

    experiment = Experiment.create("p", tmp_path)
    specs = [
        RecordingImportSpec(
            id="s1", metadata={"a": "1"}, video_parts=[_video(tmp_path / "a.mp4")]
        ),
        RecordingImportSpec(
            id="s2", metadata={"a": "2"}, video_parts=[_video(tmp_path / "b.mp4")]
        ),
    ]
    create_shells(experiment, specs)
    errors, cancelled = import_pending(experiment, specs, should_cancel=lambda: True)
    assert cancelled is True
    # both kept (tags preserved), marked cancelled — not deleted
    assert {s.id for s in experiment.recordings} == {"s1", "s2"}
    assert experiment.get_recording("s1").import_status == "cancelled"
    assert experiment.recording_metadata["s2"] == {"acquisition_setup": "", "a": "2"}


def test_import_single_and_multi_part_video(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    specs = [
        RecordingImportSpec(id="one", video_parts=[_video(tmp_path / "a.mp4")]),
        RecordingImportSpec(
            id="two",
            video_parts=[_video(tmp_path / "b1.mp4"), _video(tmp_path / "b2.mp4")],
        ),
    ]
    errors, cancelled = import_specs(experiment, specs)
    assert errors == [] and cancelled is False
    assert len(experiment.get_recording("one").video_parts) == 1
    two = experiment.get_recording("two")
    assert len(two.video_parts) == 2  # both parts, in order
    assert two.import_status == "ready"


def test_import_with_stream(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    data = tmp_path / "subject.txt"
    program = tmp_path / "task.MPC"
    data.write_text(DATA_TXT, encoding="utf-8")
    program.write_text(PROGRAM_MPC, encoding="utf-8")
    spec = RecordingImportSpec(
        id="recording_001",
        metadata={"animal_id": "M001"},
        streams=[
            StreamImportSpec(
                name="medpc",
                importer="medpc",
                raw=data,
                aux={"program": program},
                offset_ns=500,
            )
        ],
    )
    errors, cancelled = import_specs(experiment, [spec])
    assert errors == [] and cancelled is False
    recording = experiment.get_recording("recording_001")
    assert recording.import_status == "ready"
    assert [s["name"] for s in recording.streams] == ["medpc"]


def test_failed_spec_marks_failed_and_batch_continues(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    specs = [
        RecordingImportSpec(id="bad", video_parts=[tmp_path / "does_not_exist.mp4"]),
        RecordingImportSpec(id="good", video_parts=[_video(tmp_path / "ok.mp4")]),
    ]
    errors, cancelled = import_specs(experiment, specs)
    assert cancelled is False
    assert [sid for sid, _ in errors] == ["bad"]
    assert (
        experiment.get_recording("bad").import_status == "failed"
    )  # visible, deletable
    assert experiment.get_recording("good").import_status == "ready"  # batch continued


def test_cancel_keeps_completed_and_cancels_rest(tmp_path, monkeypatch):
    # cancel-after-first is inherently sequential (cancel is checked before each
    # spec) — pin K=1. Parallel cancel is covered by the pre-set-cancel test above.
    monkeypatch.setenv("DETAIL_MAX_TRANSCODE_PARALLELISM", "1")
    experiment = Experiment.create("p", tmp_path)
    specs = [
        RecordingImportSpec(id="first", video_parts=[_video(tmp_path / "a.mp4")]),
        RecordingImportSpec(id="second", video_parts=[_video(tmp_path / "b.mp4")]),
    ]
    import threading

    cancel = threading.Event()  # monotonic, like the queue's cancel flag

    def on_done(sid, ok, err):
        if sid == "first":
            cancel.set()  # cancel after the first finishes

    errors, cancelled = import_specs(
        experiment, specs, should_cancel=cancel.is_set, on_item_done=on_done
    )
    assert cancelled is True
    assert experiment.get_recording("first").import_status == "ready"  # completed, kept
    assert (
        experiment.get_recording("second").import_status == "cancelled"
    )  # kept, not run


def test_import_data_stop_cancels_and_cleans_partial(tmp_path):
    import pytest

    from detail.api.imports import create_shell, import_data
    from detail.storage.video import ImportStopped

    experiment = Experiment.create("p", tmp_path)
    spec = RecordingImportSpec(id="s1", video_parts=[_video(tmp_path / "a.mp4")])
    create_shell(experiment, spec)
    with pytest.raises(ImportStopped):  # mid-file stop propagates
        import_data(experiment, spec, should_stop=lambda: True)
    recording = experiment.get_recording("s1")
    assert recording.import_status == "cancelled"  # not "failed"
    assert recording.video_parts == []  # no part recorded
    assert not (recording.dir / "video" / "part_001.mp4").exists()  # partial removed


def test_import_pending_parallel(tmp_path, monkeypatch):
    # force >1 so import runs through the spawn process pool; every recording must
    # still import and the per-recording callbacks must fire.
    from detail.api.imports import create_shells, import_pending

    monkeypatch.setenv("DETAIL_MAX_TRANSCODE_PARALLELISM", "3")
    experiment = Experiment.create("p", tmp_path)
    specs = [
        RecordingImportSpec(
            id=f"s{i}",
            metadata={"a": str(i)},
            video_parts=[_video(tmp_path / f"v{i}.mp4")],
        )
        for i in range(3)
    ]
    create_shells(experiment, specs)
    started = []
    done = []
    progress = []
    errors, cancelled = import_pending(
        experiment,
        specs,
        on_item_started=started.append,
        on_item_done=lambda sid, ok, err: done.append((sid, ok)),
        on_progress=lambda i, t, sid, frac, status: progress.append(sid),
    )
    assert errors == [] and cancelled is False
    for i in range(3):
        recording = experiment.get_recording(f"s{i}")
        assert recording.import_status == "ready"
        assert len(recording.video_parts) == 1
    assert set(started) == {"s0", "s1", "s2"}
    assert {sid for sid, ok in done if ok} == {"s0", "s1", "s2"}
    assert progress  # worker→queue→listener→on_progress plumbing fires in parallel (D5)
    assert (
        not multiprocessing.active_children()
    )  # the spawn pool reaped its workers (D6)


def test_import_pending_parallel_failed_continues(tmp_path, monkeypatch):
    # a bad spec fails and is recorded; the good ones still import (batch continues).
    monkeypatch.setenv("DETAIL_MAX_TRANSCODE_PARALLELISM", "3")
    experiment = Experiment.create("p", tmp_path)
    specs = [
        RecordingImportSpec(id="bad", video_parts=[tmp_path / "missing.mp4"]),
        RecordingImportSpec(id="good1", video_parts=[_video(tmp_path / "a.mp4")]),
        RecordingImportSpec(id="good2", video_parts=[_video(tmp_path / "b.mp4")]),
    ]
    errors, cancelled = import_specs(experiment, specs)
    assert cancelled is False
    assert [sid for sid, _ in errors] == ["bad"]
    assert experiment.get_recording("bad").import_status == "failed"
    assert experiment.get_recording("good1").import_status == "ready"
    assert experiment.get_recording("good2").import_status == "ready"


# --------------------------------------------------------------------------- #
# Progress (§6): one recording's fraction rises through every part and phase, and the batch
# folds every recording in flight. A silent or snapping bar is indistinguishable from a hang.
# --------------------------------------------------------------------------- #


def _small_chunks(monkeypatch):
    """Copy in 256-byte chunks so a tiny fixture's copy reports more than its final byte."""
    import functools

    from detail.storage import video as video_store

    original = video_store.copy_with_progress
    monkeypatch.setattr(
        video_store,
        "copy_with_progress",
        functools.partial(original, chunk_size=256),
    )


def _stage(status):
    """The phase a status line names: ``Copying (part 2 of 2) — 1 / 2 MB`` → ``Copying``."""
    return status.split(" — ")[0].split(" (part ")[0]


def test_import_data_fraction_rises_through_every_phase_and_part(tmp_path, monkeypatch):
    """Monotonic across phases *and* parts, and moving inside each.

    The fraction used to be ``(part + stage_fraction) / parts``, so it filled on the timestamp
    scan and dropped back to the part's start for the copy — the "99% → 0%" class. Invisible
    while nothing drew a bar from it, and the first thing a bar would have shown.
    """
    from detail.api.imports import create_shell, import_data
    from detail.core import progress as progress_mod

    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    _small_chunks(monkeypatch)
    experiment = Experiment.create("p", tmp_path)
    spec = RecordingImportSpec(
        id="two",
        video_parts=[_video(tmp_path / "a.mp4"), _video(tmp_path / "b.mp4")],
    )
    create_shell(experiment, spec)
    seen: list[tuple[float, str]] = []
    import_data(experiment, spec, progress_cb=lambda f, s: seen.append((f, s)))

    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions), "the bar went backwards between phases"
    assert any(0 < f < 1 for f in fractions)
    assert fractions[-1] == pytest.approx(1.0)
    copying = [f for f, s in seen if _stage(s) == "Copying"]
    assert len(set(copying)) > 2, "the copy phase held the bar instead of moving it"
    assert any("(part 2 of 2)" in s for _, s in seen), "the line never said which part"


def test_import_data_phase_messages_advance_past_the_throttle(tmp_path, monkeypatch):
    """Every phase is named as it starts, even when the throttle would drop everything else.

    With the interval at an hour only forced emits get through, so this is exactly the set of
    lines a user sees on a fast import — and a phase that is not forced is missing from it.
    """
    from detail.api.imports import create_shell, import_data
    from detail.core import progress as progress_mod

    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 3600.0)
    experiment = Experiment.create("p", tmp_path)
    data = tmp_path / "subject.txt"
    program = tmp_path / "task.MPC"
    data.write_text(DATA_TXT, encoding="utf-8")
    program.write_text(PROGRAM_MPC, encoding="utf-8")
    spec = RecordingImportSpec(
        id="one",
        video_parts=[_video(tmp_path / "a.mp4")],
        streams=[
            StreamImportSpec(
                name="medpc", importer="medpc", raw=data, aux={"program": program}
            )
        ],
    )
    create_shell(experiment, spec)
    seen: list[str] = []
    import_data(experiment, spec, progress_cb=lambda f, s: seen.append(_stage(s)))
    phases = [p for i, p in enumerate(seen) if i == 0 or p != seen[i - 1]]
    assert phases == ["Reading timestamps", "Copying", "Importing stream medpc"]


@pytest.mark.parametrize("storage", ["default", "reference"])
def test_every_stage_add_video_part_reports_has_a_share(tmp_path, storage):
    """The import weighs a part's stages by name, so a stage it does not know holds the bar.

    A stage added to (or renamed in) ``add_video_part`` would freeze the bar for its whole
    duration with nothing failing — this is what fails instead.
    """
    from detail.api.imports import _phase_spans

    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("r")
    stages: set[str] = set()
    recording.add_video_part(
        _video(tmp_path / "videos" / "a.mp4"),
        storage=storage,
        progress=lambda stage, _f, _d: stages.add(stage),
    )
    spans = _phase_spans(storage == "reference")
    assert stages and stages <= set(spans), f"no share for {stages - set(spans)}"
    assert max(end for _, end in spans.values()) == 1.0  # the part's phases fill it


def test_parts_are_weighed_by_size_not_counted(tmp_path):
    from detail.api.imports import _part_spans

    big = tmp_path / "big.bin"
    small = tmp_path / "small.bin"
    big.write_bytes(b"x" * 300)
    small.write_bytes(b"x" * 100)
    (a, b), (c, d) = _part_spans([big, small])
    assert (a, b, c, d) == pytest.approx((0.0, 0.75, 0.75, 1.0))
    # nothing readable to weigh: fall back to counting rather than dividing by zero
    assert _part_spans([tmp_path / "gone.mp4"]) == [(0.0, 1.0)]
    assert _part_spans([]) == []  # a recording that is only streams


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _spec(sid, name=None):
    return RecordingImportSpec(id=sid, name=name or sid, video_parts=[Path("v.mp4")])


def test_fold_counts_every_recording_in_flight_and_estimates_the_rest():
    from detail.api.imports import ImportProgressFold

    clock = _Clock()
    fold = ImportProgressFold(clock=clock)
    fold.add([_spec("a", "first"), _spec("b"), _spec("c"), _spec("d")])
    fold.started("a")
    fold.started("b")
    fold.progress("a", 0.5, "Copying")
    fold.finished("c", ok=False)
    clock.now = 30.0
    event = fold.progress("b", 0.25, "Reading timestamps")
    assert event.recordings_done == 1 and event.recordings_total == 4
    assert event.failed == 1
    assert event.in_flight == (
        ("first", 0.5, "Copying"),
        ("b", 0.25, "Reading timestamps"),
    )
    assert event.fraction == pytest.approx((1 + 0.5 + 0.25) / 4)
    assert event.eta_s == pytest.approx(30.0 * (1 - 0.4375) / 0.4375)


def test_fold_ignores_a_tick_that_lands_after_its_recording_finished():
    """A parallel worker's last progress message can be drained after its result arrives."""
    from detail.api.imports import ImportProgressFold

    fold = ImportProgressFold(clock=_Clock())
    fold.add([_spec("a"), _spec("b")])
    fold.started("a")
    fold.finished("a", ok=True)
    event = fold.progress("a", 0.99, "Copying")
    assert event.in_flight == ()
    assert event.fraction == pytest.approx(0.5)
    assert fold.finished("a", ok=True).recordings_done == 1  # counted once


def test_fold_grows_when_more_is_queued_and_skips_metadata_only_recordings():
    from detail.api.imports import ImportProgressFold

    fold = ImportProgressFold(clock=_Clock())
    fold.add([_spec("a"), RecordingImportSpec(id="empty")])
    fold.finished("a", ok=True)
    event = fold.add([_spec("b")])
    assert (event.recordings_done, event.recordings_total) == (1, 2)


def test_fold_refuses_a_spec_whose_shell_does_not_exist_yet():
    from detail.api.imports import ImportProgressFold

    with pytest.raises(ValueError, match="create its shell first"):
        ImportProgressFold().add([RecordingImportSpec(name="new", video_parts=["v"])])


def test_a_single_recording_import_reports_mid_run_progress_with_an_eta(
    tmp_path, monkeypatch
):
    """The cadence test, on one recording of one part: the case the rule exists for.

    A count of recordings cannot move inside a one-recording import, so the inner fraction is
    what has to — through the real ``import_pending`` path a surface drives.
    """
    from detail.api.imports import ImportProgressFold, create_shells, import_pending
    from detail.core import progress as progress_mod

    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    monkeypatch.setenv("DETAIL_MAX_TRANSCODE_PARALLELISM", "1")
    _small_chunks(monkeypatch)
    experiment = Experiment.create("p", tmp_path)
    specs = [RecordingImportSpec(id="only", video_parts=[_video(tmp_path / "a.mp4")])]
    create_shells(experiment, specs)
    clock = _Clock()

    def tick():
        clock.now += 1.0
        return clock.now

    fold = ImportProgressFold(clock=tick)
    fold.add(specs)
    events = []
    import_pending(
        experiment,
        specs,
        on_progress=lambda _i, _n, sid, f, s: events.append(fold.progress(sid, f, s)),
        on_item_done=lambda sid, ok, _e: events.append(fold.finished(sid, ok)),
    )
    mid = [e for e in events if 0 < e.fraction < 1]
    assert mid, "no event between 0 and 1 on a single-recording import"
    assert all(e.eta_s is not None for e in mid)
    assert events[-1].fraction == 1.0 and events[-1].recordings_done == 1
