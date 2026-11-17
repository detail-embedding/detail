"""Experiment export / import — slim/full round-trip, collision, integrity (§15), and progress.

**The progress half of a recorded class** (#916; the caller half is
``tests/test_progress_callers.py``): *a long operation whose only feedback is that it started and
that it finished.* Packing an experiment is two full passes over its payload (a sha256, then the
tar), and on a real one that is 30,588 files and 17 GB without its video. The catalogue's bundle
emitted ``packing 0%`` before that one call and ``packing 100%`` after it, so an export run sat at
zero for the whole pack; the Experiment page, the CLI and Home's Import reported nothing at all.

**How it hid.** Nothing is wrong at any line: the archive is correct, the import succeeds. And
the obvious test passes against the bug. A fixture of many files moves a bar that only steps per
file, so a test on it says nothing about the single large zarr shard, or video, that holds the
bar at one value for most of the pack. The cadence tests here use **one** large file and assert
two reports *inside* it.

**The rules this file holds:**

- Every pass reports from inside a file: chunk credit while hashing, a metered reader while
  packing, archive bytes consumed while extracting.
- The fraction is weighed by what each file costs each pass, measured per file and per byte
  (`storage/export.py::_PassCost`), not by counting files or bytes alone, so a linear ETA over it
  tells the truth. gzip packing gets the share its cost earns.
- Import reports its last before it replaces an existing experiment, so a callback that raises to
  cancel never lands between the delete and the rename, and a cancel leaves no staging folder.

**A second class, found by the first fix** (#916): *an archive whose digest describes different
bytes than it holds.* An export reads the payload twice, a sha256 then the tar, and a file rewritten
between the reads is packed under a digest that no longer matches it. Nothing notices at export: the
import extracts every member, 17 GB of them, and then fails its integrity check. Making the bar move
made it certain. An export *run* keeps its record in ``exports/<run_id>/``, inside the experiment it
packs, and every progress report rewrites it, so every bundle made by a run failed on import. The
tests here called the pack directly, outside a run, where that record does not exist; the workflow
test is ``tests/api/test_export_run.py::test_a_bundle_made_by_an_export_run_imports``. The first
instance of the class was an archive written inside the tree it packed.

- ``exports/`` is not packed: what is rewritten during every export is left out, not raced.
- Every other file is compared, by ``fstat`` of the open handle, as hashed, before it is packed and
  after; a change stops the export with ``ExperimentChangedError`` naming the file.
- The archive is written as ``.partial`` and renamed when complete, so a stopped export leaves
  nothing that looks finished.

**A third class** (#916): *a name that means something only on the machine that minted it,
carried to another machine without its meaning.* A recording whose video is referenced in place
names it by a video-root alias, which resolves through that machine's ``storage.yaml``, because an
experiment file may hold no machine path (Tenet 6). An imported experiment named roots the
importing machine had never registered, so every recording's video was missing, though both
machines reached the same server; locating one file by hand registered the root and fixed them
all. **How it hid:** every round-trip test exported and imported on the one ``storage.yaml``, where
each alias was already registered, so the missing fact was never missing.

- The manifest's ``video_roots`` says where each alias the recordings name pointed on the
  exporting machine. A machine path in the archive's manifest, never in an experiment file.
- Import registers a root found there only when the first referenced file in it matches its
  recorded fingerprint, the evidence a Locate gives, and no less.
- Then it looks **beside every folder that held video**. The first real study tried kept its
  video as a folder per day, and only the one day-folder the importing PC already knew was
  found: the manifest is no help when the archive predates it or the two PCs spell the server
  differently (an IP on one, a name on the other). The layouts are `_LAYOUTS`; a share that does
  not answer is asked once, not once per day-folder.
- An alias this machine already uses for another folder is never re-pointed; the imported
  recordings are pointed at a root of this machine's own.
- The search only reads, and what it found is applied after the last report, so a cancel leaves
  this machine's roots as they were. The tests stand on two ``storage.yaml``s (`_machine`).
"""

import io
import tarfile

import cv2
import numpy as np
import pytest
import yaml

from detail.api.experiment import Experiment
from detail.storage import datasets as datasets_store
from detail.storage import export as export_store
from detail.storage import models as models_store


def _video(path, n=4, w=16, h=16):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), (i * 7) % 250 + 1, np.uint8))
    vw.release()
    return path


def _experiment(root):
    """A experiment with a video recording + metadata + a model + a dataset."""
    proj = Experiment.create("study", root)
    proj.create_recording("recording_001", metadata={"animal_id": "M001"})
    proj.get_recording("recording_001").add_video_part(_video(root / "_src.mp4"))
    mdir = models_store.model_dir(proj.dir, "model_001")
    (mdir / "run").mkdir(parents=True)
    (mdir / "run" / "best.pt").write_bytes(b"weights")
    models_store.write_model_yaml(
        mdir,
        {"id": "model_001", "status": "complete", "weights": {"best": "run/best.pt"}},
    )
    ddir = datasets_store.dataset_dir(proj.dir, "dataset_001")
    (ddir / "images").mkdir(parents=True)
    (ddir / "images" / "a.jpg").write_bytes(b"img")
    datasets_store.write_dataset_yaml(ddir, {"id": "dataset_001", "status": "complete"})
    return proj


def test_export_import_slim_roundtrip(tmp_path):
    proj = _experiment(tmp_path / "a")
    archive = proj.export(tmp_path / "study.tar")  # slim default
    manifest = export_store.read_manifest(archive)
    assert manifest["id"] == "study"
    assert manifest["contents"] == {
        "video_included": False,
        "video_requested": False,
        "referenced_video": [],
        "datasets_included": False,
        "models_included": True,
        "symlinked_folders_left_out": [],
    }
    ws = tmp_path / "ws"
    ws.mkdir()
    dest = export_store.import_experiment(archive, ws)
    assert (dest / "config.yaml").is_file()
    imported = Experiment(dest)
    assert imported.recording_metadata["recording_001"] == {"animal_id": "M001"}
    sdir = dest / "recordings" / "recording_001"
    assert not list((sdir / "video").glob("*.mp4"))  # video excluded (slim)
    ts = sdir / "video" / "timestamps" / "generated" / "part_001.csv"
    assert ts.is_file()  # timestamps kept (needed to read detections)
    assert (dest / "models" / "model_001" / "model.yaml").is_file()  # models in
    assert not (dest / "datasets" / "dataset_001").exists()  # datasets excluded


def test_export_full_includes_video_and_datasets(tmp_path):
    proj = _experiment(tmp_path / "a")
    archive = proj.export(
        tmp_path / "full.tar", include_video=True, include_datasets=True
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    dest = export_store.import_experiment(archive, ws)
    assert list((dest / "recordings" / "recording_001" / "video").glob("*.mp4"))
    assert (dest / "datasets" / "dataset_001" / "images" / "a.jpg").is_file()


def test_import_collision_requires_replace(tmp_path):
    proj = _experiment(tmp_path / "a")
    archive = proj.export(tmp_path / "s.tar")
    ws = tmp_path / "ws"
    ws.mkdir()
    export_store.import_experiment(archive, ws)
    with pytest.raises(FileExistsError):
        export_store.import_experiment(archive, ws)
    export_store.import_experiment(archive, ws, replace=True)  # ok with replace
    other = export_store.import_experiment(archive, ws, name="study_copy")
    assert other.name == "study_copy" and other.is_dir()  # separate copy


def test_portability_guard_rejects_absolute_path(tmp_path):
    proj = _experiment(tmp_path / "a")
    sy = proj.dir / "recordings" / "recording_001" / "recording.yaml"
    sy.write_text(
        sy.read_text(encoding="utf-8-sig") + "\nleak: /Users/someone/clip.mp4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absolute paths"):
        proj.export(tmp_path / "x.tar")


def _write_stream_file(proj, stream, subdir, name, text):
    path = (
        proj.dir / "recordings" / "recording_001" / "streams" / stream / subdir / name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# A BORIS export records the annotator's own media path; §3 stores raw files verbatim,
# so scanning them made every experiment carrying one permanently un-exportable.
_BORIS_ROW = "obs,Source,Behavior\nx,player #1:/Users/someone/part_001.mp4,climb\n"


def test_portability_guard_ignores_importer_raw_stream_files(tmp_path):
    proj = _experiment(tmp_path / "a")
    _write_stream_file(proj, "climb_rear", "raw", "boris.csv", _BORIS_ROW)
    archive = proj.export(tmp_path / "x.tar")
    ws = tmp_path / "ws"
    ws.mkdir()
    dest = export_store.import_experiment(archive, ws)
    raw = dest / "recordings" / "recording_001" / "streams" / "climb_rear" / "raw"
    assert (
        raw.joinpath("boris.csv").read_text(encoding="utf-8-sig") == _BORIS_ROW
    )  # carried through verbatim


def test_portability_guard_still_rejects_canonical_stream_csv(tmp_path):
    """The DETAIL-written half of a stream stays under the guard."""
    proj = _experiment(tmp_path / "a")
    _write_stream_file(proj, "climb_rear", "canonical", "part_001.csv", _BORIS_ROW)
    with pytest.raises(ValueError, match="absolute paths"):
        proj.export(tmp_path / "x.tar")


def test_portability_guard_not_fooled_by_stream_named_raw(tmp_path):
    """The carve-out is the ``raw/`` subdir, not any path segment spelled 'raw'."""
    proj = _experiment(tmp_path / "a")
    _write_stream_file(proj, "raw", "canonical", "part_001.csv", _BORIS_ROW)
    with pytest.raises(ValueError, match="absolute paths"):
        proj.export(tmp_path / "x.tar")


def test_import_rejects_unsafe_name(tmp_path):
    proj = _experiment(tmp_path / "a")
    archive = proj.export(tmp_path / "s.tar")
    ws = tmp_path / "ws"
    ws.mkdir()
    for bad in ("../evil", "a/b", ".."):
        with pytest.raises(ValueError, match="invalid experiment name"):
            export_store.import_experiment(archive, ws, name=bad)
    assert not list(ws.iterdir())  # nothing placed


def test_import_detects_corruption(tmp_path):
    proj = _experiment(tmp_path / "a")
    archive = proj.export(tmp_path / "s.tar")
    bad = tmp_path / "bad.tar"
    with tarfile.open(archive, "r:*") as tin:  # repack with a wrong payload checksum
        manifest = yaml.safe_load(tin.extractfile("export.yaml").read())
        manifest["payload_sha256"] = "0" * 64
        blob = yaml.dump(manifest, sort_keys=False).encode()
        with tarfile.open(bad, "w") as tout:
            info = tarfile.TarInfo("export.yaml")
            info.size = len(blob)
            tout.addfile(info, io.BytesIO(blob))
            for m in tin.getmembers():
                if m.name == "export.yaml":
                    continue
                tout.addfile(m, tin.extractfile(m) if m.isfile() else None)
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError, match="integrity"):
        export_store.import_experiment(bad, ws)


# --------------------------------------------------------------------------- #
# the manifest says what is inside, not what was asked for
# --------------------------------------------------------------------------- #
def test_the_manifest_reports_video_that_was_packed_not_video_that_was_requested(
    tmp_path,
):
    """**The class.** *A manifest that records the request instead of the result.*

    `video_included` was the `include_video` argument. Those are the same thing only while
    every video lives inside the experiment — and a **reference-in-place** part does not: it
    sits under a machine-local video root, outside the tree `_included_files` walks, so no
    setting can pack it. A recipient reading `video_included: true` and finding no video has
    been told something untrue by the one file whose job is to say what is inside.

    Nothing fails when it happens: the tar is valid, the import succeeds, and every frame view
    is simply dead on the other machine. The same shape as a status written before the work
    that justifies it.
    """
    from detail.storage import recording as recording_store

    proj = _experiment(tmp_path / "a")
    directory = proj.get_recording("recording_001").dir
    # Re-point the part DETAIL copied at a video root instead, and drop the copy: the shape a
    # reference-in-place import leaves behind.
    data = recording_store.read_recording(directory)
    for part in (data.get("video") or {}).get("files", []):
        copied = directory / part.pop("file", "")
        copied.unlink(missing_ok=True)
        part["source"] = {"root": "rig", "rel": "day1/clip.mp4"}
    recording_store.write_recording(directory, data)

    archive = proj.export(tmp_path / "study.tar", include_video=True)
    contents = export_store.read_manifest(archive)["contents"]

    assert contents["video_requested"] is True, "the request is still recorded"
    assert (
        contents["video_included"] is False
    ), "the manifest claims video the bundle does not contain"
    assert contents["referenced_video"] == [
        "recording_001"
    ], "the recipient is not told which recordings' video could not travel"


def test_referenced_video_recordings_finds_only_the_referenced_ones(tmp_path):
    """Guard the guard: a copied part must not be reported as referenced, or the list is noise."""
    from detail.storage import recording as recording_store

    proj = _experiment(tmp_path / "a")
    proj.create_recording("recording_002")
    for name, part in (
        ("recording_001", {"source": {"root": "rig", "rel": "a.mp4"}}),
        ("recording_002", {"file": "video/part_001.mp4"}),
    ):
        directory = proj.get_recording(name).dir
        data = recording_store.read_recording(directory)
        data["video"] = {"files": [part]}
        recording_store.write_recording(directory, data)

    assert export_store.referenced_video_recordings(proj.dir) == ["recording_001"]


# --------------------------------------------------------------------------- #
# progress (#916): both directions report, and the bar moves inside one file
# --------------------------------------------------------------------------- #
#: Larger than the read chunk several times over, so one file is several reads.
_BIG = 4 * export_store._CHUNK + 123


def _solo(root, size=_BIG):
    """An experiment whose payload is one large file beside its config.

    One of everything the passes loop over, as the cadence rule asks: a test with many files
    passes on the file count alone, while the single-file case (one big zarr shard, or one
    video) is the one that sat at 0% for the whole pack.
    """
    proj = Experiment.create("solo", root)
    blob = proj.dir / "analysis" / "blob.bin"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(np.random.default_rng(0).bytes(size))
    return proj


def _every_emit(monkeypatch):
    from detail.core import progress as progress_mod

    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    events = []
    return events, lambda stage, fraction, detail: events.append(
        (stage, fraction, detail)
    )


def _stages(events) -> list[str]:
    """The stages in the order they were reported, each once."""
    out = []
    for stage, _fraction, _detail in events:
        if not out or out[-1] != stage:
            out.append(stage)
    return out


def _moved_inside_one_file(events, stage) -> bool:
    """Two reports of ``stage`` with the same files-done count and a rising fraction.

    The per-file credit changes the count, so only a report from *inside* a file can repeat it
    with more done.
    """
    seen: dict[str, float] = {}
    for got, fraction, detail in events:
        if got != stage:
            continue
        if detail in seen and fraction > seen[detail]:
            return True
        seen.setdefault(detail, fraction)
    return False


def test_export_reports_each_phase_in_order_and_ends_at_one(tmp_path, monkeypatch):
    events, progress = _every_emit(monkeypatch)
    _solo(tmp_path / "a").export(tmp_path / "solo.tar", progress=progress)

    assert _stages(events) == [
        export_store.STAGE_SCANNING,
        export_store.STAGE_CHECKING,
        export_store.STAGE_HASHING,
        export_store.STAGE_PACKING,
    ]
    fractions = [fraction for _stage, fraction, _detail in events]
    assert fractions == sorted(fractions), "the bar went backwards"
    assert events[-1][:2] == (export_store.STAGE_PACKING, 1.0)
    assert any(0 < f < 1 for f in fractions)


@pytest.mark.parametrize(
    "stage", [export_store.STAGE_HASHING, export_store.STAGE_PACKING]
)
def test_one_large_file_moves_the_export_bar_while_it_is_read(
    tmp_path, monkeypatch, stage
):
    """Revert the chunk credit in either pass and its bar jumps a whole file at a time."""
    events, progress = _every_emit(monkeypatch)
    _solo(tmp_path / "a").export(tmp_path / "solo.tar", progress=progress)
    assert _moved_inside_one_file(events, stage), [e for e in events if e[0] == stage]


def test_a_gzip_export_gives_packing_the_share_it_costs(tmp_path, monkeypatch):
    """The bar is weighed by what each pass costs, not split evenly between them.

    Plain, hashing and packing cost about the same, so packing starts near halfway. gzip is
    many times slower to pack, so the same export must reach packing far earlier, or its ETA (which
    extrapolates from this fraction) says minutes when it means an hour.
    """
    proj = _solo(tmp_path / "a")

    def packing_starts_at(gzip: bool) -> float:
        events, progress = _every_emit(monkeypatch)
        proj.export(tmp_path / f"solo_{gzip}.tar", gzip=gzip, progress=progress)
        return next(f for s, f, _d in events if s == export_store.STAGE_PACKING)

    plain, compressed = packing_starts_at(False), packing_starts_at(True)
    assert 0.3 < plain < 0.7, plain
    assert compressed < 0.15, compressed


def test_the_bar_counts_small_files_as_well_as_bytes(tmp_path, monkeypatch):
    """Many small files beside one large one: neither count alone tracks the time.

    Measured, a file costs about what 680 KB of its bytes do. Weighed by bytes alone, 300 small
    files would be worth ~0 of the hashing bar and it would sit still through them; this asserts
    they are worth a real share of it.
    """
    proj = _solo(tmp_path / "a")
    small = proj.dir / "analysis" / "chunks"
    small.mkdir()
    for index in range(300):
        (small / f"{index:03d}").write_bytes(b"x" * 64)
    events, progress = _every_emit(monkeypatch)
    proj.export(tmp_path / "solo.tar", progress=progress)

    hashing = [(f, d) for s, f, d in events if s == export_store.STAGE_HASHING]
    start, end = hashing[0][0], hashing[-1][0]
    # Files are hashed in path order, so the small ones are everything after the large one.
    order = export_store._included_files(
        proj.dir, include_video=False, include_datasets=False
    )
    assert order.index("analysis/blob.bin") < order.index("analysis/chunks/000")
    big_done = order.index("analysis/blob.bin") + 1
    after_big = next(f for f, d in hashing if d.startswith(f"{big_done} /"))
    assert (end - after_big) / (end - start) > 0.5


def test_import_reports_each_phase_and_finishes_before_it_places(tmp_path, monkeypatch):
    archive = _solo(tmp_path / "a").export(tmp_path / "solo.tar")
    ws = tmp_path / "ws"
    ws.mkdir()
    events, progress = _every_emit(monkeypatch)
    export_store.import_experiment(archive, ws, progress=progress)

    assert _stages(events) == [
        export_store.STAGE_READING,
        export_store.STAGE_EXTRACTING,
        export_store.STAGE_VERIFYING,
        export_store.STAGE_PLACING,
    ]
    fractions = [fraction for _stage, fraction, _detail in events]
    assert fractions == sorted(fractions), "the bar went backwards"
    assert events[-1][:2] == (export_store.STAGE_PLACING, 1.0)


@pytest.mark.parametrize("gzip", [False, True], ids=["plain", "gzip"])
@pytest.mark.parametrize(
    "stage", [export_store.STAGE_EXTRACTING, export_store.STAGE_VERIFYING]
)
def test_one_large_member_moves_the_import_bar(tmp_path, monkeypatch, gzip, stage):
    """Extraction is metered by archive bytes consumed, so it moves inside a member, and
    through a gzipped archive the same way."""
    archive = _solo(tmp_path / "a").export(tmp_path / "solo.tar", gzip=gzip)
    ws = tmp_path / "ws"
    ws.mkdir()
    events, progress = _every_emit(monkeypatch)
    export_store.import_experiment(archive, ws, progress=progress)
    assert _moved_inside_one_file(events, stage), [e for e in events if e[0] == stage]


@pytest.mark.parametrize(
    "stage",
    [
        export_store.STAGE_EXTRACTING,
        export_store.STAGE_VERIFYING,
        export_store.STAGE_PLACING,
    ],
)
def test_a_cancelled_import_places_nothing_and_leaves_no_staging(
    tmp_path, monkeypatch, stage
):
    """A callback that raises (the GUI's Cancel) stops the import with nothing behind it.

    The staging folder of an import killed mid-extract was found in a real workspace at 18.7 GB;
    a cancel must not be a second way to leave one. And an existing experiment of the same id
    is untouched, since the report that could raise comes before the replace, never between
    the delete and the rename.
    """
    from detail.core.progress import OperationCancelled

    archive = _solo(tmp_path / "a").export(tmp_path / "solo.tar")
    ws = tmp_path / "ws"
    ws.mkdir()
    existing = export_store.import_experiment(archive, ws)
    (existing / "keep.txt").write_text("mine", encoding="utf-8")
    _every_emit(monkeypatch)

    def cancel_at(got, fraction, detail):
        if got == stage and not detail.startswith("0 /"):
            raise OperationCancelled()

    with pytest.raises(OperationCancelled):
        export_store.import_experiment(archive, ws, replace=True, progress=cancel_at)
    assert sorted(p.name for p in ws.iterdir()) == ["solo"]
    assert (existing / "keep.txt").read_text(encoding="utf-8-sig") == "mine"


def test_an_archive_written_inside_the_experiment_is_not_packed_into_itself(tmp_path):
    """`tar.add` skipped the archive it was writing, after the digest had hashed it.

    So a second export to the same path inside the tree wrote a bundle that failed its own
    integrity check on import. It is now left out of both passes.
    """
    proj = _solo(tmp_path / "a", size=1024)
    inside = proj.dir / "solo.tar"
    proj.export(inside)
    proj.export(inside)  # the first archive now exists inside the tree it packs
    with tarfile.open(inside) as tar:
        assert f"{export_store._ROOT}/solo.tar" not in tar.getnames()
    ws = tmp_path / "ws"
    ws.mkdir()
    export_store.import_experiment(inside, ws)  # the digest matches what was packed


def test_an_earlier_archive_saved_in_the_experiment_is_not_packed_into_later_ones(
    tmp_path,
):
    """**A fourth class** (#29): *an export that packs what it left behind before.* Only the
    archive being written was dropped, so ``--out study/first.tar`` and then any later export
    carried the first archive, and each export after that carried all the ones before it.

    An archive is recognised by name and confirmed by its manifest, so a ``.tar`` of the user's
    own that is not a DETAIL export still travels; a ``.partial`` one is an export never
    finished.
    """
    proj = _solo(tmp_path / "a", size=1024)
    proj.export(proj.dir / "first.tar")
    proj.export(proj.dir / "first.tar.gz", gzip=True)
    (proj.dir / "stopped.tar.partial").write_bytes(b"half an archive")
    (proj.dir / "mine.tar").write_bytes(b"the user's own, not an export")

    second = proj.export(tmp_path / "second.tar")

    with tarfile.open(second) as tar:
        packed = {
            name.removeprefix(f"{export_store._ROOT}/")
            for name in tar.getnames()
            if "/" in name and "/" not in name.removeprefix(f"{export_store._ROOT}/")
        }
    assert packed == {
        "config.yaml",
        "mine.tar",
    }, "an earlier archive was packed, or the user's own .tar was not"
    ws = tmp_path / "ws"
    ws.mkdir()
    export_store.import_experiment(second, ws)  # the digest matches what was packed


def test_a_symlinked_folder_left_out_is_named_in_the_manifest(tmp_path):
    """A symlinked folder is not followed, and a symlinked file is: true before, but said
    nowhere, so the recipient found the folder missing. The manifest names each one, for the
    CLI to print (`tests/test_cli.py`)."""
    import os

    proj = _solo(tmp_path / "a", size=1024)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "notes.txt").write_text("x", encoding="utf-8")
    try:
        os.symlink(elsewhere, proj.dir / "linked", target_is_directory=True)
    except OSError:
        pytest.skip("this machine cannot make a symlink")

    archive = proj.export(tmp_path / "s.tar")

    assert export_store.read_manifest(archive)["contents"][
        "symlinked_folders_left_out"
    ] == ["linked"]
    with tarfile.open(archive) as tar:
        assert not any("linked" in name for name in tar.getnames())


def test_reading_the_manifest_stops_at_the_manifest(tmp_path, monkeypatch):
    """`getmember` loads every header first: a whole decompression pass on a gzipped bundle,
    before an import has reported anything. The manifest is the first member; read it there.
    """
    archive = _solo(tmp_path / "a").export(tmp_path / "solo.tar", gzip=True)

    def every_member(_self):
        raise AssertionError("read every member to find the first one")

    monkeypatch.setattr(tarfile.TarFile, "getmembers", every_member)
    assert export_store.read_manifest(archive)["id"] == "solo"


# --------------------------------------------------------------------------- #
# the two passes read the same bytes (#916): an export run packed its own job.json
# --------------------------------------------------------------------------- #
def _rewrite_when(stage, path, events):
    """A progress callback that changes ``path`` the first time ``stage`` is reported."""
    done = []

    def progress(got, fraction, detail):
        events.append((got, fraction, detail))
        if got == stage and not done:
            done.append(True)
            with open(path, "ab") as handle:
                handle.write(b"rewritten")

    return progress


def _no_archive(dest) -> bool:
    return not dest.exists() and not dest.with_name(dest.name + ".partial").exists()


def test_export_run_records_are_not_packed(tmp_path):
    """An export run's record lives in ``exports/`` and is rewritten by its every report.

    So it was hashed in one state and packed in another, and the bundle failed its integrity
    check on import after extracting every member. The records describe copies this machine
    wrote elsewhere, under root names another machine does not have, so none of them travel.
    """
    proj = _solo(tmp_path / "a", size=1024)
    record = proj.dir / "exports" / "export_001" / "job.json"
    record.parent.mkdir(parents=True)
    record.write_text('{"status": "running"}', encoding="utf-8")

    archive = proj.export(tmp_path / "solo.tar")

    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert not [n for n in names if n.startswith(f"{export_store._ROOT}/exports/")]
    assert (
        f"{export_store._ROOT}/analysis/blob.bin" in names
    ), "the fixture packed nothing"


def test_a_file_changed_between_the_passes_stops_the_export(tmp_path, monkeypatch):
    """Something writing to the experiment between the digest and the pack — another run, the
    app — would ship a bundle that fails on import, after a 17 GB extract. It stops here instead,
    naming the file, and leaves no archive behind."""
    events, _ = _every_emit(monkeypatch)
    proj = _solo(tmp_path / "a", size=1024)
    blob = proj.dir / "analysis" / "blob.bin"
    dest = tmp_path / "solo.tar"

    with pytest.raises(export_store.ExperimentChangedError, match="analysis/blob.bin"):
        proj.export(
            dest, progress=_rewrite_when(export_store.STAGE_PACKING, blob, events)
        )
    assert _no_archive(dest)


def test_a_file_changed_while_it_is_packed_stops_the_export(tmp_path, monkeypatch):
    """The same check after the copy: a write landing mid-file is a torn member."""
    events, _ = _every_emit(monkeypatch)
    proj = _solo(tmp_path / "a")
    blob = proj.dir / "analysis" / "blob.bin"
    dest = tmp_path / "solo.tar"
    packing_inside_blob = []

    def progress(stage, fraction, detail):
        events.append((stage, fraction, detail))
        # A report from inside the blob's copy: packing has begun and the count has not moved.
        if stage == export_store.STAGE_PACKING and fraction > 0.5:
            packing_inside_blob.append(detail)
            if len(packing_inside_blob) == 2:
                with open(blob, "ab") as handle:
                    handle.write(b"rewritten")

    with pytest.raises(export_store.ExperimentChangedError, match="analysis/blob.bin"):
        proj.export(dest, progress=progress)
    assert _no_archive(dest)


def test_a_cancelled_export_leaves_no_archive(tmp_path, monkeypatch):
    """Written under ``.partial`` and renamed when complete: a stopped pack used to leave a
    truncated ``.tar`` carrying the name of a finished one."""
    from detail.core.progress import OperationCancelled

    _every_emit(monkeypatch)
    dest = tmp_path / "solo.tar"

    def cancel_while_packing(stage, fraction, detail):
        if stage == export_store.STAGE_PACKING and fraction > 0.6:
            raise OperationCancelled()

    with pytest.raises(OperationCancelled):
        _solo(tmp_path / "a").export(dest, progress=cancel_while_packing)
    assert _no_archive(dest)


# --------------------------------------------------------------------------- #
# video referenced in place is found where the exporting machine had it (#916)
# --------------------------------------------------------------------------- #
def _machine(monkeypatch, directory):
    """Stand on another machine: its own ``storage.yaml``, so its own video roots."""
    from detail.storage import locations

    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(locations, "CONFIG_DIR", directory)
    monkeypatch.setattr(locations, "STORAGE_PATH", directory / "storage.yaml")


def _exported_from_a_server(tmp_path, monkeypatch):
    """Two recordings whose video sits on a server under two roots, exported on machine A.

    Returns standing on machine B, which reaches the same server and has never registered
    either root: the shape of a lab moving an experiment between two PCs. Two roots, so a test
    can stop an import between them.
    """
    from detail.storage import video_roots

    _machine(monkeypatch, tmp_path / "machine_a")
    server = tmp_path / "server"
    proj = Experiment.create("study", tmp_path / "a")
    for i, alias in enumerate(("lab", "rig"), start=1):
        video = _video(server / alias / f"day{i}" / "clip.mp4", n=3 + i)
        video_roots.register_video_root(alias, server / alias)
        proj.create_recording(f"recording_00{i}")
        proj.get_recording(f"recording_00{i}").add_video_part(
            video, storage="reference", video_root=alias
        )
    archive = proj.export(tmp_path / "study.tar")
    _machine(monkeypatch, tmp_path / "machine_b")
    ws = tmp_path / "ws"
    ws.mkdir()
    return archive, server, ws


def _sources(experiment_dir):
    from detail.storage import recording as recording_store

    return {
        directory.name: part["source"]
        for directory in sorted((experiment_dir / "recordings").iterdir())
        for part in recording_store.read_recording(directory)["video"]["files"]
    }


def test_an_import_finds_video_where_the_exporting_machine_had_it(
    tmp_path, monkeypatch
):
    """**A third class** (#916): *a name that means something only on the machine that minted
    it, carried to another machine without its meaning.*

    A recording names its referenced video by a root alias, and the alias resolves through this
    machine's ``storage.yaml``: an experiment file may hold no machine path (Tenet 6). So an
    imported experiment named roots the importing machine had never heard of, and every
    recording's video was missing, though both machines reached the same server. Locating one
    file by hand registered the root and fixed them all, which is the evidence the tool had
    been missing: where the root was. The archive is the one thing that crosses machines, so it
    carries that, and the import looks there.
    """
    from detail.storage import video_roots

    archive, server, ws = _exported_from_a_server(tmp_path, monkeypatch)
    assert export_store.read_manifest(archive)["video_roots"] == {
        alias: (server / alias).resolve().as_posix() for alias in ("lab", "rig")
    }
    assert video_roots.list_video_roots() == {}, "machine B already knew the roots"
    events, progress = _every_emit(monkeypatch)

    dest = export_store.import_experiment(archive, ws, progress=progress)

    assert video_roots.list_video_roots() == {
        alias: (server / alias).resolve() for alias in ("lab", "rig")
    }
    assert Experiment(dest).referenced_video_missing() == []
    assert _sources(dest) == {
        "recording_001": {"root": "lab", "rel": "day1/clip.mp4"},
        "recording_002": {"root": "rig", "rel": "day2/clip.mp4"},
    }, "a root that was free here was renamed anyway"
    assert _stages(events)[-2:] == [
        export_store.STAGE_FINDING_VIDEO,
        export_store.STAGE_PLACING,
    ]


@pytest.mark.parametrize(
    "holder", [None, "nas"], ids=["a_root_of_its_own", "a_root_that_holds_it"]
)
def test_a_root_name_this_machine_uses_elsewhere_is_never_repointed(
    tmp_path, monkeypatch, holder
):
    """Auto-registered roots are named after their folder, so two machines can each have a
    ``lab`` that is not the same folder. Re-pointing this machine's would break every experiment
    here that uses it; the imported recordings are pointed at a root of this machine's own.
    """
    from detail.storage import video_roots

    archive, server, ws = _exported_from_a_server(tmp_path, monkeypatch)
    elsewhere = tmp_path / "local" / "lab"
    elsewhere.mkdir(parents=True)
    video_roots.register_video_root("lab", elsewhere)
    if holder:
        video_roots.register_video_root(holder, server)

    dest = export_store.import_experiment(archive, ws)

    assert video_roots.video_root("lab") == elsewhere.resolve()
    assert Experiment(dest).referenced_video_missing() == []
    expected = (
        {"root": "nas", "rel": "lab/day1/clip.mp4"}
        if holder
        else {"root": "lab_2", "rel": "day1/clip.mp4"}
    )
    assert _sources(dest) == {
        "recording_001": expected,
        "recording_002": {"root": "rig", "rel": "day2/clip.mp4"},
    }


@pytest.mark.parametrize("what", ["replaced", "unreachable"])
def test_a_folder_without_the_same_video_is_not_taken(tmp_path, monkeypatch, what):
    """The evidence a user's Locate gives, and no less: a file at the recorded path whose
    fingerprint matches. A same-named different video, or a server this machine cannot reach,
    registers nothing, and the import still succeeds."""
    import shutil

    from detail.storage import video_roots

    archive, server, ws = _exported_from_a_server(tmp_path, monkeypatch)
    if what == "replaced":
        for alias, i in (("lab", 1), ("rig", 2)):
            _video(server / alias / f"day{i}" / "clip.mp4", n=9)
    else:
        shutil.rmtree(server)

    dest = export_store.import_experiment(archive, ws)

    assert video_roots.list_video_roots() == {}
    assert Experiment(dest).referenced_video_missing() == [
        "recording_001",
        "recording_002",
    ]


def test_a_different_video_under_a_root_name_this_machine_uses_is_not_found(
    tmp_path, monkeypatch
):
    """**The third class, from the other side** (#29): *a name that means something only on the
    machine that minted it* was trusted on this machine when this machine also used the name.

    Machine B already has its own ``lab`` root, holding a different ``day1/clip.mp4``, and
    cannot reach A's server. The import rightly does not take B's folder (its file fails the
    fingerprint), but the recording kept resolving to it: ``referenced_video_missing`` and
    ``video_available`` asked only whether a file was there, so nothing was listed, and every
    preview, frame read and clip export showed B's animal against A's timestamps. The run gate
    always refused it; the readers did not. Now the resolver checks the recorded fingerprint, so
    the recording is listed as not found and a frame read refuses rather than reading it.
    """
    import shutil

    from detail.storage import video_roots

    archive, server, ws = _exported_from_a_server(tmp_path, monkeypatch)
    shutil.rmtree(server)  # A's folders are not reachable from B
    here = tmp_path / "local" / "lab"
    _video(here / "day1" / "clip.mp4", n=9, w=24, h=24)
    video_roots.register_video_root("lab", here)

    dest = export_store.import_experiment(archive, ws)

    imported = Experiment(dest)
    assert video_roots.video_root("lab") == here.resolve(), "B's root was re-pointed"
    assert imported.referenced_video_missing() == [
        "recording_001",
        "recording_002",
    ], "a different video at the same path counted as found"
    recording = imported.get_recording("recording_001")
    assert recording.video_available is False
    with pytest.raises(
        video_roots.VideoNotRecordedError, match="not the video recorded"
    ):
        recording.get_frame(0, 0)
    issues = recording.verify_videos()
    assert [i.kind for i in issues] == [
        "changed_fingerprint"
    ], "the run gate's own report changed"


def test_a_referenced_file_replaced_at_the_same_path_is_checked_again(tmp_path):
    """The resolver's fingerprint is cached per state of the file, so a reader resolving on
    every frame pays a ``stat``. A file replaced at the same path is a new state: a cache keyed
    on the path alone would keep vouching for the video that was there before."""
    from detail.storage import recording as recording_store
    from detail.storage import video_roots

    folder = tmp_path / "videos"
    video = _video(folder / "clip.mp4", n=4)
    proj = Experiment.create("study", tmp_path / "a")
    proj.create_recording("recording_001")
    recording = proj.get_recording("recording_001")
    recording.add_video_part(video, storage="reference")
    part = recording_store.read_recording(recording.dir)["video"]["files"][0]
    assert (
        recording_store.resolve_part_video_path(recording.dir, part) == video.resolve()
    )

    _video(video, n=9, w=24, h=24)

    with pytest.raises(video_roots.VideoNotRecordedError):
        recording_store.resolve_part_video_path(recording.dir, part)
    assert recording.video_available is False


@pytest.mark.parametrize(
    "stage", [export_store.STAGE_FINDING_VIDEO, export_store.STAGE_PLACING]
)
def test_a_cancelled_import_registers_no_video_root(tmp_path, monkeypatch, stage):
    """The search only reads, and what it found is registered after the last report, so a
    cancel at any report leaves this machine's roots as they were, as well as the workspace.
    Stopped on the *second* root, so a search that registered the first as it went is caught.
    """
    from detail.core.progress import OperationCancelled
    from detail.storage import video_roots

    archive, _server, ws = _exported_from_a_server(tmp_path, monkeypatch)
    _every_emit(monkeypatch)

    def cancel_at(got, _fraction, detail):
        if got == stage and not detail.startswith("1 /"):
            raise OperationCancelled()

    with pytest.raises(OperationCancelled):
        export_store.import_experiment(archive, ws, progress=cancel_at)
    assert video_roots.list_video_roots() == {}
    assert list(ws.iterdir()) == []


# --------------------------------------------------------------------------- #
# ...and beside the folders this machine knows: a study is a folder per day (#916)
# --------------------------------------------------------------------------- #
#: Where each root's folder is, under the study's folder, and which one the importing machine
#: already knows. Auto-named roots are named after their folder, deduplicated with ``_2``.
_LAYOUTS = {
    # One folder per day, side by side: the layout that was reported.
    "siblings": ({"Day1": "Day1", "Day2": "Day2", "Day3": "Day3"}, "Day1"),
    # The same folder registered twice on the exporting machine (two spellings of the server).
    "same_folder_twice": ({"Day1": "Day1", "Day1_2": "Day1"}, "Day1"),
    # Named Day2_2 because the exporting machine's Day2 was some other folder.
    "a_suffixed_name_only": ({"Day1": "Day1", "Day2_2": "Day2"}, "Day1"),
    # Videos in the study folder and in a folder under it.
    "a_folder_inside": ({"study": ".", "Day2": "Day2"}, "study"),
    # The known folder is inside the one that is missing.
    "the_folder_around": ({"study": ".", "Day2": "Day2"}, "Day2"),
    # Found only from a folder that was itself found: Day1 around cam1, then Day2 beside Day1.
    "one_step_at_a_time": (
        {"cam1": "Day1/cam1", "Day1": "Day1", "Day2": "Day2"},
        "cam1",
    ),
}


def _exported_by_day(tmp_path, monkeypatch, layout, manifest):
    """A study with a folder per day on a server, exported on machine A; returns on machine B,
    which knows one of the folders and is handed an archive whose ``video_roots`` it cannot use.
    """
    from detail.storage import video_roots

    folders, known = _LAYOUTS[layout]
    study = tmp_path / "server" / "study"
    _machine(monkeypatch, tmp_path / "machine_a")
    proj = Experiment.create("study", tmp_path / "a")
    for i, (alias, folder) in enumerate(folders.items(), start=1):
        video = _video(study / folder / f"clip{i}.mp4", n=3 + i)
        video_roots.register_video_root(alias, (study / folder).resolve())
        proj.create_recording(f"recording_00{i}")
        proj.get_recording(f"recording_00{i}").add_video_part(
            video, storage="reference", video_root=alias
        )
    real = export_store._video_root_locations
    if manifest == "made_before_it_said":
        monkeypatch.setattr(export_store, "_video_root_locations", lambda _d: {})
    else:  # a spelling of the server this machine does not answer to
        monkeypatch.setattr(
            export_store,
            "_video_root_locations",
            lambda d: {a: (tmp_path / "gone" / a).as_posix() for a in real(d)},
        )
    archive = proj.export(tmp_path / "study.tar")
    _machine(monkeypatch, tmp_path / "machine_b")
    video_roots.register_video_root(known, (study / folders[known]).resolve())
    ws = tmp_path / "ws"
    ws.mkdir()
    return archive, study, folders, ws


@pytest.mark.parametrize("manifest", ["made_before_it_said", "says_a_path_not_here"])
@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_the_other_folders_are_found_beside_the_one_this_machine_knows(
    tmp_path, monkeypatch, layout, manifest
):
    """Reported on a real study: video in a folder per day, ``Day1`` known on the
    importing PC, ``Day2`` … ``Day5`` and two ``_2`` duplicates not found.
    The manifest's paths are no help when the archive predates them or the two PCs spell the
    server differently; the folder the importing PC does know is."""
    from detail.storage import video_roots

    archive, study, folders, ws = _exported_by_day(
        tmp_path, monkeypatch, layout, manifest
    )

    dest = export_store.import_experiment(archive, ws)

    assert Experiment(dest).referenced_video_missing() == []
    assert {alias: video_roots.video_root(alias) for alias in folders} == {
        alias: (study / folder).resolve() for alias, folder in folders.items()
    }


def test_nothing_is_guessed_without_a_folder_to_start_from(tmp_path, monkeypatch):
    """A machine that knows none of the folders has nothing to look beside: no root is
    registered, and every recording is named as not found."""
    from detail.storage import locations
    from detail.storage import video_roots

    archive, _study, _folders, ws = _exported_by_day(
        tmp_path, monkeypatch, "siblings", "made_before_it_said"
    )
    locations.STORAGE_PATH.unlink()

    dest = export_store.import_experiment(archive, ws)

    assert video_roots.list_video_roots() == {}
    assert len(Experiment(dest).referenced_video_missing()) == 3


def test_a_guessed_folder_without_the_same_video_is_not_taken(tmp_path, monkeypatch):
    """A guess is a guess: the folder beside is used only when a video in it matches."""
    from detail.storage import video_roots

    archive, study, _folders, ws = _exported_by_day(
        tmp_path, monkeypatch, "siblings", "made_before_it_said"
    )
    _video(study / "Day2" / "clip2.mp4", n=9)  # same name, different video

    dest = export_store.import_experiment(archive, ws)

    assert video_roots.video_root("Day2") is None
    assert video_roots.video_root("Day3") == (study / "Day3").resolve()
    assert Experiment(dest).referenced_video_missing() == ["recording_002"]


def test_a_share_that_does_not_answer_is_asked_once(tmp_path, monkeypatch):
    """A server name only the exporting machine resolves makes each lookup wait out the network
    timeout. Asked per root, that was one wait per day-folder; it is asked once per share.
    """
    import pathlib

    archive, _study, _folders, ws = _exported_by_day(
        tmp_path, monkeypatch, "siblings", "says_a_path_not_here"
    )
    asked = []
    real_is_dir = pathlib.Path.is_dir

    def is_dir(self):
        if self.anchor and self == pathlib.Path(self.anchor):
            asked.append(self)
            return False  # the share does not answer
        return real_is_dir(self)

    monkeypatch.setattr(pathlib.Path, "is_dir", is_dir)

    export_store.import_experiment(archive, ws)

    assert len(asked) == 1, asked


# --------------------------------------------------------------------------- #
# Replace says what it deletes and the bundle does not bring back (#29)
# --------------------------------------------------------------------------- #
def _with_an_export_record(proj):
    from detail.storage import experiment as experiment_store

    record = experiment_store.exports_dir(proj.dir) / "export_01"
    record.mkdir(parents=True)
    (record / "export.yaml").write_text("id: export_01\n", encoding="utf-8")


def test_replace_lists_what_a_slim_bundle_does_not_bring_back(tmp_path):
    """**A fifth class** (#29): *an import that replaces more than it brings.* Replace swaps the
    whole folder, so the copied video and datasets a slim bundle leaves out, and the export
    records no bundle carries, were deleted without a word: the loop ``docs/cli.md`` describes
    for moving an experiment back and forth lost the video on the way home.

    The owner's decision: delete as today, but say what first. One function decides the list
    (`export_store.replace_losses`), it rides on the refusal so the GUI asks without walking the
    experiment on its UI thread, and one function words it (`api.experiment.replace_warning`);
    the CLI and the GUI prompt each render those lines (`tests/test_cli.py`,
    `tests/gui/test_export.py`).
    """
    from detail.api.experiment import replace_warning

    proj = _experiment(tmp_path / "a")
    _with_an_export_record(proj)
    (video,) = (proj.dir / "recordings").rglob("*.mp4")
    archive = proj.export(tmp_path / "study.tar")  # slim: no video, no datasets

    with pytest.raises(export_store.ExperimentExistsError) as refused:
        export_store.import_experiment(archive, proj.dir.parent)

    assert isinstance(
        refused.value, FileExistsError
    ), "callers catching it stop catching"
    assert refused.value.losses == export_store.ReplaceLosses(
        experiment_id="study",
        video_files=1,
        video_bytes=video.stat().st_size,
        datasets=("dataset_001",),
        export_records=("export_01",),
    )
    assert replace_warning(refused.value.losses) == [
        f"1 video file(s), {video.stat().st_size} B: the bundle carries no video",
        "1 dataset(s), dataset_001: the bundle carries no datasets",
        "1 export record(s), export_01: no bundle carries them (the exported files "
        "themselves are not touched)",
    ]

    export_store.import_experiment(archive, proj.dir.parent, replace=True)

    assert not video.exists(), "Replace no longer deletes what it said it would"
    assert not (proj.dir / "datasets" / "dataset_001").exists()


def test_a_bundle_that_carries_video_and_datasets_lists_neither(tmp_path):
    """The negative case: the list is read from what the bundle packed, so a full bundle,
    which brings the video and datasets back, lists only what no bundle carries."""
    proj = _experiment(tmp_path / "a")
    _with_an_export_record(proj)
    archive = proj.export(
        tmp_path / "full.tar", include_video=True, include_datasets=True
    )

    with pytest.raises(export_store.ExperimentExistsError) as refused:
        export_store.import_experiment(archive, proj.dir.parent)

    assert refused.value.losses == export_store.ReplaceLosses(
        experiment_id="study", export_records=("export_01",)
    )


def test_an_earlier_archive_in_the_experiment_is_listed_as_lost(tmp_path):
    """No export packs an earlier archive, so a replace deletes one saved inside the
    experiment for good."""
    proj = _solo(tmp_path / "a", size=16)
    proj.export(proj.dir / "first.tar")
    archive = proj.export(tmp_path / "second.tar")

    with pytest.raises(export_store.ExperimentExistsError) as refused:
        export_store.import_experiment(archive, proj.dir.parent)

    assert refused.value.losses == export_store.ReplaceLosses(
        experiment_id="solo", archives=("first.tar",)
    )
    assert refused.value.losses.empty is False
