"""An import that fails half-way leaves nothing behind (#29 Part 5).

**The class: an import that writes before it has finished checking.** Each of these made its
first write, then ran a check (or a long copy) that could still fail, and on the failure kept
what it had written:

- a spatial CSV wrote its regions, then refused a point outside ``[0, 1]``; the retry, after
  fixing the file, refused the regions as "already exists";
- ``add_stream`` copied the files into ``streams/<name>/raw/`` and wrote the canonical CSV
  before its last checks, so a refused import kept a folder with no manifest entry, and a retry
  with another file left the first in ``raw/`` for good;
- ``add-stream --save-preset`` added the stream, then refused the taken preset name, so the
  command failed with the stream already in and a re-run said the stream existed;
- ``model import`` (and promoting a distilled model) copied the weights before writing
  ``model.yaml``, so a cut-short import left a folder that blocked the name, that no listing
  showed and that ``model remove`` refused as "no imported model";
- an example-set import made the set, then read its images, so an unreadable PNG left an empty
  set.

It hid because every one of them raised a clean, correct error: the test that the refusal
happens passed, and nothing looked at the disk afterwards. **The rule** (Tenet 10, nothing
placed on a validation error): check everything before the first write, or build aside and
move into place only once the last check has passed (``core.atomic.staged_dir``), or remove
what was made on any failure, Ctrl-C included. **This guard** snapshots the tree an import
writes into, makes the import fail at its last step, and asserts the tree is byte-for-byte
what it was.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from detail.api import models as api_models
from detail.api.experiment import Experiment
from detail.cli import main
from detail.core.dataclasses import StreamImportResult
from detail.core.protocols import StreamImporter
from detail.core.registries import STREAM_IMPORTERS, TRAINED_LOADERS
from detail.models import distilled
from detail.models import imported as imported_store
from detail.storage import experiment as experiment_store
from detail.storage import models as models_store

#: An id spent on a failed import stays spent (ids are never reused, #27), so the mark that
#: records it may move; it is bookkeeping about ids, not a piece of the import.
_ID_BOOKKEEPING = {".ids.yaml", ".ids.lock"}


def _tree(root: Path) -> dict[str, bytes | None]:
    """Every file and folder under ``root``: a file's bytes, ``None`` for a folder."""
    return {
        p.relative_to(root).as_posix(): (None if p.is_dir() else p.read_bytes())
        for p in sorted(Path(root).rglob("*"))
        if p.name not in _ID_BOOKKEEPING
    }


def test_the_tree_snapshot_would_catch_an_empty_leftover_folder(tmp_path):
    """The comparison sees a folder with nothing in it, the smallest leftover there is."""
    before = _tree(tmp_path)
    (tmp_path / "streams" / "pose").mkdir(parents=True)
    assert _tree(tmp_path) != before


def _square(lo: float = 0.1, hi: float = 0.5):
    return [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]


# --- spatial CSV ----------------------------------------------------------------------- #


def test_a_spatial_csv_with_one_bad_point_writes_no_region(tmp_path):
    exp = Experiment.create("proj", tmp_path)
    rec = exp.create_recording("r1")
    csv = tmp_path / "setup.csv"
    csv.write_text(
        "kind,name,active_start,active_end,frame_idx,vertex_index,x,y\n"
        "roi,zone,,,0,0,0.1,0.1\nroi,zone,,,0,1,0.5,0.1\nroi,zone,,,0,2,0.5,0.5\n"
        "point,lever,,,0,0,1.2,0.5\n",
        encoding="utf-8",
    )
    before = _tree(rec.dir)

    with pytest.raises(
        ValueError, match=r"vertex \(1\.2, 0\.5\) at frame 0 is outside"
    ):
        rec.import_spatial_csv(csv)

    assert _tree(rec.dir) == before
    # The retry the leftover used to block: fix the point, and both arrive.
    fixed = csv.read_text(encoding="utf-8-sig").replace("1.2,0.5", "0.9,0.5")
    csv.write_text(fixed, encoding="utf-8")
    assert rec.import_spatial_csv(csv) == (["zone", "lever"], [], [])


# --- add_stream ------------------------------------------------------------------------ #


def _pose_csv(tmp_path: Path) -> Path:
    path = tmp_path / "pose.csv"
    path.write_text("t_sec,nose_x\n0,1\n0.1,3\n", encoding="utf-8")
    return path


_TIME_CONFIG = {"time_column": "t_sec", "time_kind": "seconds"}


def test_a_stream_refused_at_its_last_check_leaves_no_folder(tmp_path):
    """``covered_behaviors`` is checked after the files are copied and the canonical CSV is
    written: the last refusal before the manifest, so the widest window for a leftover.
    """
    exp = Experiment.create("proj", tmp_path)
    rec = exp.create_recording("r1")
    before = _tree(rec.dir)

    with pytest.raises(ValueError, match="covered_behaviors only applies"):
        rec.add_stream(
            "pose",
            _pose_csv(tmp_path),
            "timeseries_csv",
            config=_TIME_CONFIG,
            covered_behaviors=["a"],
        )

    assert _tree(rec.dir) == before


class _InterruptedImporter(StreamImporter):
    """Spills a file into its output folder, then is stopped (Ctrl-C) mid-parse."""

    importer_id = "interrupted"
    display_name = "Interrupted"
    stream_type = "events"
    role = None
    clock = "stream_seconds"
    aux_inputs = []

    def can_import(self, path: Path) -> bool:
        return True

    def import_stream(self, data_path, out_dir, aux_paths, config=None):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "spill.tmp").write_text("half", encoding="utf-8")
        raise KeyboardInterrupt


def test_a_stream_import_stopped_mid_parse_leaves_no_folder(tmp_path, monkeypatch):
    monkeypatch.setitem(STREAM_IMPORTERS, "interrupted", _InterruptedImporter)
    exp = Experiment.create("proj", tmp_path)
    rec = exp.create_recording("r1")
    before = _tree(rec.dir)

    with pytest.raises(KeyboardInterrupt):
        rec.add_stream("pose", _pose_csv(tmp_path), "interrupted")

    assert _tree(rec.dir) == before


def test_a_folder_left_by_an_older_failed_import_does_not_mix_into_the_retry(tmp_path):
    """A leftover made before imports were staged is no stream's: the retry replaces it,
    rather than keeping its file in ``raw/`` beside the new one."""
    exp = Experiment.create("proj", tmp_path)
    rec = exp.create_recording("r1")
    leftover = rec.dir / "streams" / "pose" / "raw"
    leftover.mkdir(parents=True)
    (leftover / "first_try.csv").write_text("t_sec,nose_x\n0,9\n", encoding="utf-8")

    rec.add_stream("pose", _pose_csv(tmp_path), "timeseries_csv", config=_TIME_CONFIG)

    assert sorted(p.name for p in leftover.iterdir()) == ["pose.csv"]
    assert [s["name"] for s in rec.streams] == ["pose"]


def test_add_stream_with_a_taken_preset_name_adds_nothing(tmp_path):
    """``--save-preset`` saves after the import; its name is checked before it."""
    exp = Experiment.create("proj", tmp_path)
    rec = exp.create_recording("r1")
    exp.save_stream_preset("dlc", "timeseries_csv", _TIME_CONFIG)
    before = _tree(exp.dir)

    result = CliRunner().invoke(
        main,
        # fmt: off
        [
            "recording", "add-stream", "--experiment", str(exp.dir), "--recording", "r1",
            "--name", "pose", "--importer", "timeseries_csv",
            "--data", str(_pose_csv(tmp_path)),
            "--time-column", "t_sec", "--time-is", "seconds", "--save-preset", "dlc",
        ],
        # fmt: on
    )

    assert result.exit_code == 1, result.output
    assert "a stream preset named 'dlc' already exists" in result.output
    # The fix it names is a command that exists, not an `overwrite` this command lacks.
    assert "detail experiment delete-stream-preset dlc" in result.output
    assert "overwrite" not in result.output
    assert _tree(exp.dir) == before
    assert rec.streams == []


# --- model catalog --------------------------------------------------------------------- #


class _StubLoader:
    """A trained-loader stand-in whose ``introspect`` reads no real checkpoint."""

    @staticmethod
    def introspect(weights_path):
        return ["mouse"], "detect"


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """The catalog with one good model already in it, so a failure is seen to leave the
    neighbours alone as well as to add nothing."""
    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubLoader)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    imported_store.import_model(weights, family="stubfam", name="neighbour")
    return weights


def test_a_model_import_stopped_after_the_weights_copy_leaves_nothing(
    catalog, tmp_path, monkeypatch
):
    preview = tmp_path / "shot.png"
    preview.write_bytes(b"\x89PNG")
    before = _tree(imported_store.catalog_dir())

    def stopped(model_dir, image_path):
        raise KeyboardInterrupt

    monkeypatch.setattr(imported_store, "set_preview", stopped)
    with pytest.raises(KeyboardInterrupt):
        imported_store.import_model(
            catalog, family="stubfam", name="mine", preview=preview
        )

    assert _tree(imported_store.catalog_dir()) == before
    assert imported_store.discover() == ["neighbour"]


def test_a_model_import_with_a_missing_preview_is_refused_before_the_copy(
    catalog, tmp_path
):
    before = _tree(imported_store.catalog_dir())

    with pytest.raises(FileNotFoundError, match="preview image not found"):
        imported_store.import_model(
            catalog, family="stubfam", name="mine", preview=tmp_path / "nope.png"
        )

    assert _tree(imported_store.catalog_dir()) == before


def test_a_folder_left_by_an_older_model_import_can_be_removed(catalog):
    """Made before imports were staged: no ``model.yaml``, so not a model, but it held the
    name. The refusal says how to clear it, and ``remove`` can."""
    leftover = imported_store.model_dir("mine")
    (leftover / "weights").mkdir(parents=True)
    (leftover / imported_store.WEIGHTS_RELPATH).write_bytes(b"half")

    with pytest.raises(ValueError, match="detail model remove mine"):
        imported_store.import_model(catalog, family="stubfam", name="mine")
    api_models.remove_imported_model("mine")

    assert not leftover.exists()
    assert imported_store.import_model(catalog, family="stubfam", name="mine") == "mine"
    # Only a name the catalog lists is deleted: an id is never a path.
    with pytest.raises(FileNotFoundError, match="no imported model"):
        api_models.remove_imported_model("..")
    assert imported_store.catalog_dir().is_dir()


def test_a_promotion_that_fails_after_the_weights_copy_leaves_nothing(
    catalog, tmp_path
):
    """The preview is copied after the weights; one recorded in a sub-folder fails there."""
    proj = experiment_store.create_experiment(tmp_path, "rp")
    mdir = models_store.model_dir(proj, "model_001")
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "best.pt").write_bytes(b"weights-bytes")
    (mdir / "shots").mkdir()
    (mdir / "shots" / "preview.png").write_bytes(b"\x89PNG")
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "mouse_seg",
            "model_family": "yolo",
            "task": "segment",
            "classes": [{"id": 0, "name": "mouse"}],
            "weights": {"best": "run/weights/best.pt"},
            "status": "complete",
            "preview": "shots/preview.png",
        },
    )
    before = _tree(imported_store.catalog_dir())

    with pytest.raises(FileNotFoundError):
        distilled.register_distilled_model(
            proj, "model_001", source_experiment_name="rp", source_experiment_id="rp"
        )

    assert _tree(imported_store.catalog_dir()) == before


# --- example sets ---------------------------------------------------------------------- #


def test_an_example_set_with_an_unreadable_image_leaves_no_set(tmp_path):
    """The bad image is the last of two, so the first was already copied in."""
    import cv2
    import numpy as np

    exp = Experiment.create("proj", tmp_path / "exp")
    kept = exp.import_example_set(_skeleton_file(tmp_path))
    src = tmp_path / "shared"
    src.mkdir()
    cv2.imwrite(str(src / "good.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    (src / "bad.png").write_bytes(b"not a png")
    (src / "examples.yaml").write_text(
        _set_yaml(["good.png", "bad.png"]), encoding="utf-8"
    )
    before = _tree(exp.dir)

    with pytest.raises(FileNotFoundError, match="missing or unreadable"):
        exp.import_example_set(src / "examples.yaml")

    assert _tree(exp.dir) == before
    assert [s.id for s in exp.example_sets()] == [kept.id]


def _skeleton_file(tmp_path: Path) -> Path:
    path = tmp_path / "rat.yaml"
    path.write_text("names: [nose, tail]\nedges: [[nose, tail]]\n", encoding="utf-8")
    return path


def _set_yaml(images: list[str]) -> str:
    import yaml

    frame = {
        "recording_id": "r1",
        "tracking_run_id": "track_01",
        "track_id": 0,
        "part_id": "part_001",
        "frame_idx": 0,
        "t_ns": 0,
        "bbox": [0, 0, 8, 8],
        "marks": [],
    }
    return yaml.safe_dump(
        {
            "schema_version": 1,
            "name": "shared",
            "purpose": "example",
            "skeleton": {"names": ["nose", "tail"], "edges": [["nose", "tail"]]},
            "frames": [{**frame, "image": image} for image in images],
        }
    )
