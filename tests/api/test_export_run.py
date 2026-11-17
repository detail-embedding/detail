"""An export is a run: one request, one record, one folder (#354).

**What this file is really about.** `write_export` has its own tests; these drive it through the
executor — a real worker subprocess, a real run dir, a real reattach — because that is the part
that was not true before and the part a user notices when they close the laptop.

Two properties carry the design:

- **One request is one run**, however many targets it covers. Three targets are three *units* of
  one run with one folder and one manifest, not three runs. The failure this guards would look
  perfectly reasonable on the Active-runs page, which is why it is asserted rather than assumed.
- **The destination never reaches an experiment file as a path.** Everything in a job lands in
  the run record via `runners.job_params`; a record is `.yaml`; `storage/export.py` scans those
  and *refuses the bundle* on `/Users/`. An absolute destination would therefore not fail here —
  it would brick the experiment the next time somebody exported it.

**A class guarded here: a file numbered by its position in a filtered list (#26).** The video
target skips parts referenced in place, and numbered what it copied by position among the rest,
so with part 1 referenced, part 2 and 3's videos came out as ``part001`` / ``part002`` beside the
timestamps target's ``part002`` / ``part003``. Each file now carries its part's own number
(``test_exported_video_files_carry_their_parts_own_number``).
"""

from __future__ import annotations

import pytest

from detail.api import export_targets  # noqa: F401  (registers the catalogue)
from detail.api import exporters
from detail.api.experiment import Experiment
from detail.storage import export_runs as export_runs_store


def _run(cohort, targets, out, **kwargs):
    handle = cohort.experiment.run_export(
        "analysis_run", "an_01", targets, out, **kwargs
    )
    return handle.wait()


def test_one_request_is_one_run(authored_cohort, tmp_path):
    """Three targets, one run — the property the whole design turns on."""
    out = tmp_path / "out"
    handle = _run(authored_cohort, ["instances", "ethogram"], out)

    assert handle.status == "complete"
    assert export_runs_store.discover_exports(authored_cohort.experiment.dir) == [
        handle.id
    ], "one export run, not one per target"

    folders = list(out.iterdir())
    assert len(folders) == 1, f"one folder, not one per target: {folders}"
    assert (folders[0] / exporters.MANIFEST_NAME).is_file()

    record = authored_cohort.experiment.export_record(handle.id)
    assert record["target_ids"] == ["instances", "ethogram"]
    assert record["status"] == "complete"
    assert "instances.csv" in record["files"]


def test_the_run_is_listed_while_it_is_going_and_after(authored_cohort, tmp_path):
    """It reaches `export_runs`, which is what `api.active` reads for the Active-runs row."""
    handle = _run(authored_cohort, ["instances"], tmp_path / "out")
    refs = authored_cohort.experiment.export_runs

    assert [r.id for r in refs] == [handle.id]
    assert refs[0].target_ids == ["instances"]
    assert refs[0].created_at, (
        "an export ref without `created_at` shows a blank 'started' column and sorts last — "
        "the wart the other four refs have"
    )


def test_the_destination_never_lands_in_an_experiment_file(authored_cohort, tmp_path):
    """The guard whose absence bricks an experiment rather than failing a test.

    An absolute path in the job reaches the record through `runners.job_params` and the record
    is scanned by the bundle exporter, which then refuses **permanently**. So: assert the record
    holds an alias, and then assert the thing that actually matters — the bundle still builds.
    """
    out = tmp_path / "somewhere" / "deep"
    handle = _run(authored_cohort, ["instances"], out)

    run_dir = export_runs_store.export_dir(authored_cohort.experiment.dir, handle.id)
    text = (run_dir / export_runs_store.EXPORT_FILENAME).read_text(encoding="utf-8-sig")
    assert str(tmp_path) not in text, "the record holds a machine path"
    assert "/Users/" not in text and "/home/" not in text

    record = authored_cohort.experiment.export_record(handle.id)
    assert set(record["destination"]) == {"root", "rel"}, "an alias, not a path"

    # The real consequence, asserted end to end rather than inferred from the scan's rules.
    bundle = tmp_path / "bundle.tar"
    authored_cohort.experiment.export(bundle)
    assert bundle.is_file()


def test_the_worker_writes_the_outcome_of_every_target(authored_cohort, tmp_path):
    """A run finishes with nobody watching, so what the message box used to say is recorded."""
    handle = _run(authored_cohort, ["instances", "ethogram"], tmp_path / "out")
    record = authored_cohort.experiment.export_record(handle.id)

    outcomes = {o["target"]: o for o in record["outcomes"]}
    assert set(outcomes) == {"instances", "ethogram"}
    assert all(o["status"] == "complete" for o in outcomes.values())
    assert outcomes["instances"][
        "files"
    ], "a target's own files are named, not just the run's"


def test_a_finished_run_reattaches_from_a_fresh_process(authored_cohort, tmp_path):
    """The point of the whole change: the record outlives the process that launched it."""
    handle = _run(authored_cohort, ["instances"], tmp_path / "out")

    reopened = Experiment.open(authored_cohort.experiment.dir)
    assert reopened.attach_export_run(handle.id).status == "complete"
    assert reopened.attach_active_export_run(handle.id) is None, "it is not still going"


def test_a_run_launched_with_clip_options_can_read_its_own_record_back(
    authored_cohort, tmp_path
):
    """**The class.** *A value that survives one of a job's two writers and not the other.*

    A job goes to ``job.json`` through ``json.dump``, which turns a tuple into a list without
    comment, **and** to the run record through ``runners.job_params`` + ``yaml.dump``, which
    writes ``!!python/tuple`` — which ``yaml.safe_load`` then refuses. Every other job hand-lists
    typed fields; this one takes an opaque `options` bag from a surface, and the GUI's clip knobs
    put a tuple in `renders`.

    What made it bad rather than cosmetic: the **runner reads its own record** to write the
    outcome, so the export did the work and then died — envelope `failed`, files on disk, and a
    listing that silently dropped the whole experiment because `export_runs` raised. Reported
    from the app on the first GUI export with clips ticked.

    Asserted through the real launch path and the real reader, not on `to_dict` alone: the bug
    was in the seam between them.
    """
    handle = authored_cohort.experiment.run_export(
        "analysis_run",
        "an_01",
        ["instances"],
        tmp_path / "out",
        renders=("overlay", "mask"),  # a tuple, exactly as the dialog builds it
        crop=True,
        size=320,
    )
    handle.wait()

    record = authored_cohort.experiment.export_record(handle.id)
    assert record["status"] == "complete", "the runner could not finish its own record"
    assert record["params"]["options"]["renders"] == ["overlay", "mask"]

    # The listing must not raise either — one unreadable record used to empty the whole page.
    assert [r.id for r in authored_cohort.experiment.export_runs] == [handle.id]


def test_an_unknown_target_is_refused_before_a_worker_is_spawned(
    authored_cohort, tmp_path
):
    """Validation at submit, not in the worker — an error you see now beats one in a log."""
    with pytest.raises(exporters.ExportError):
        authored_cohort.experiment.run_export(
            "analysis_run", "an_01", ["nope"], tmp_path / "out"
        )
    assert export_runs_store.discover_exports(authored_cohort.experiment.dir) == []


def test_a_target_that_cannot_run_here_is_refused_with_its_reason(
    authored_cohort, tmp_path
):
    """`clips` needs an embedding run the fixture does not record."""
    with pytest.raises(exporters.ExportError, match="embedding"):
        authored_cohort.experiment.run_export(
            "analysis_run", "an_01", ["clips"], tmp_path / "out"
        )


def test_a_renamed_folder_is_reflected_in_the_record(
    authored_cohort, tmp_path, monkeypatch
):
    """**The class.** *A name minted before the work runs, never reconciled with what ran (#475).*

    `run_export` mints the destination folder name at submit time and writes it into the record
    immediately. `write_export`'s `_unique_dir` can rename that folder if it already exists on
    disk — two exports of the same subject minted within the same wall-clock second collide in
    practice — and returns the real path as `ExportResult.dest`. The worker used to drop that on
    the floor: the record kept the pre-collision name forever, so `exporters.destination_of`
    (what both the export list and this run's own detail page use to say "where are my files")
    pointed at a folder that either belonged to the *other* export or did not exist at all, while
    `status`/`outcomes`/`files` correctly described real files sitting one directory over.
    """
    out = tmp_path / "out"
    monkeypatch.setattr(exporters, "folder_name", lambda subject, when=None: "collide")

    first = _run(authored_cohort, ["instances"], out)
    second = _run(authored_cohort, ["instances"], out)
    assert first.status == "complete"
    assert second.status == "complete"

    first_record = authored_cohort.experiment.export_record(first.id)
    second_record = authored_cohort.experiment.export_record(second.id)
    assert first_record["folder"] == "collide"
    assert (
        second_record["folder"] == "collide_2"
    ), "the record still names the pre-collision folder"

    destination = exporters.destination_of(second_record)
    assert destination == out / "collide_2"
    assert destination.is_dir()
    assert (destination / exporters.MANIFEST_NAME).is_file()


def test_a_bundle_made_by_an_export_run_imports(authored_cohort, tmp_path, monkeypatch):
    """**The workflow, not the function.** An export run keeps its record in
    ``exports/<run_id>/`` — inside the experiment it is packing — and rewrites it on every
    report, so a bundle made this way hashed the record in one state and packed it in another,
    and every import of it failed its integrity check after extracting every member (#916).
    Every bundle test called the pack directly, outside a run, where that record does not
    exist. A stage change is forced through the run's throttle, so the record is rewritten
    between the hash and the pack on every run, however small.
    """
    handle = _run_experiment(authored_cohort, ["bundle"], tmp_path / "out")
    record = authored_cohort.experiment.export_record(handle.id)
    assert record["status"] == "complete", record.get("error")

    (archive,) = (tmp_path / "out").rglob("*.tar")
    # Import places into the managed workspace, which the suite does not isolate: without this
    # the copy lands in the developer's real `DetailExperiments` (it did, once).
    from detail.storage import locations

    monkeypatch.setattr(locations, "DEFAULT_ROOT", tmp_path / "workspace")
    imported = Experiment.import_archive(archive, name="imported_copy")
    assert imported.dir.parent == tmp_path / "workspace"
    assert imported.dir.name == "imported_copy"
    assert not (imported.dir / "exports").exists(), "an export record travelled"


def _run_experiment(cohort, targets, out):
    return cohort.experiment.run_export("experiment", "", targets, out).wait()


def test_the_time_left_is_the_running_targets_own(monkeypatch):
    """An experiment bundle beside its metadata CSV is given half the bar, so a whole-run
    extrapolation read the bundle's remaining three times over: 20 minutes, with 7 to go (#916).
    The ETA is the running target's own (`PhaseClock`)."""
    from detail.core import progress as progress_mod
    from detail.pipeline import export_run

    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    now = [100.0]
    events = []
    report = export_run._reporter(events.append, 2, clock=lambda: now[0])
    report("packing", 0.0)
    now[0] = 110.0
    report(
        "packing", 0.25, "1 / 2 files"
    )  # the first of two targets, half done in 10 s

    assert events[-1].eta_s == pytest.approx(10.0), "not the bundle's own 10 s left"


def test_a_new_stage_is_reported_at_once(monkeypatch):
    """The throttle drops a report that comes too soon after the last; a change of stage must not
    be one of them, or the page says "hashing" through the start of the pack."""
    from detail.core import progress as progress_mod
    from detail.pipeline import export_run

    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 3600.0)
    events = []
    report = export_run._reporter(events.append, 1)
    report("hashing", 0.1, "a")
    report("hashing", 0.2, "b")  # inside the interval: dropped
    report("packing", 0.5, "c")  # inside the interval, but a new stage

    assert [e.message for e in events] == ["hashing · a", "packing · c"]


def test_exported_video_files_carry_their_parts_own_number(tmp_path, monkeypatch):
    """Part 1 referenced in place, parts 2 and 3 copied in. The video target skips part 1,
    and the files it writes must still say part 2 and part 3, matching the timestamps
    target's files for the same parts (#26). Before the fix they were numbered by position
    among the copied parts, so part 3's video sat beside part 2's timestamps."""
    import csv

    import cv2
    import numpy as np

    from detail.storage import locations

    monkeypatch.setattr(locations, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")
    proj = Experiment.create("p", tmp_path / "exp")
    recording = proj.create_recording("s1")
    for n, storage in ((5, "reference"), (6, "copy"), (7, "copy")):
        src = tmp_path / "external" / f"clip_{n}.mp4"
        src.parent.mkdir(exist_ok=True)
        vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24))
        for _ in range(n):
            vw.write(np.zeros((24, 32, 3), np.uint8))
        vw.release()
        if storage == "reference":
            recording.add_video_part(src, storage="reference")
        else:
            recording.add_video_part(src)
    assert [bool(p.get("source")) for p in recording.video_parts] == [
        True,
        False,
        False,
    ]

    subject = exporters.ExportSubject("recording", proj.dir, recording.id)
    result = exporters.write_export(
        subject,
        [
            exporters.target("recording", "video"),
            exporters.target("recording", "timestamps"),
        ],
        tmp_path / "out",
    )
    videos = sorted(f for f in result.files if f.endswith(".mp4"))
    stamps = sorted(f for f in result.files if f.endswith("_timestamps.csv"))
    assert videos == [f"{recording.id}_part002.mp4", f"{recording.id}_part003.mp4"]
    assert stamps == [f"{recording.id}_part{i:03d}_timestamps.csv" for i in (1, 2, 3)]
    # The pairing a user makes by name is the right one: part 3's video has part 3's frames.
    cap = cv2.VideoCapture(str(result.dest / f"{recording.id}_part003.mp4"))
    stamps_csv = result.dest / f"{recording.id}_part003_timestamps.csv"
    with open(stamps_csv, encoding="utf-8-sig") as handle:
        n_stamps = len(list(csv.DictReader(handle)))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n_stamps
    cap.release()
