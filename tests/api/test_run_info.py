"""A run's meta-information: every YAML at the top of its directory, as on disk (``api/run_info``).

What is guarded:

- **The files a reader needs, and only those.** The record and the module's own file first, the
  rest alphabetically; nothing from the folders under the run (profiles, caches are results) and
  nothing that is not YAML (``job.json``, CSVs).
- **The text as written**, not a re-dump: a comment survives, and key order is the file's.
- **An export that holds every file shown**, under a folder named for the run, and leaves no
  ``.tmp`` behind.
"""

from __future__ import annotations

import zipfile

import pytest

from detail.api import run_info
from detail.api.experiment import Experiment
from detail.storage import analysis_runs as analysis_store

MODULE_TEXT = (
    "# kept as written\nmethod: codebook_viterbi\nmethod_params:\n  codewords: 700\n"
)
MACHINE_TEXT = "schema_version: 1\ncpu_cores: 16\n"
RESOURCES_TEXT = "schema_version: 1\npeak_ram_bytes: 1510567936\n"


def a_run(tmp_path, run_id: str = "an_01") -> Experiment:
    """An experiment with one analysis run whose directory holds the files a real one does."""
    experiment = Experiment.create("meta", tmp_path)
    run_dir = analysis_store.analysis_dir(experiment.dir, run_id)
    analysis_store.write_analysis_yaml(
        run_dir,
        {
            "id": run_id,
            "module_id": "temporal_clustering",
            "recording_ids": [],
            "status": "failed",
            "params": {"window_s": 0.53},
        },
    )
    (run_dir / "temporal_clustering.yaml").write_text(MODULE_TEXT, encoding="utf-8")
    (run_dir / "resources.yaml").write_text(RESOURCES_TEXT, encoding="utf-8")
    (run_dir / "machine.yaml").write_text(MACHINE_TEXT, encoding="utf-8")
    (run_dir / "job.json").write_text("{}", encoding="utf-8")
    (run_dir / "activity.csv").write_text("a,b\n", encoding="utf-8")
    (run_dir / "profiles").mkdir()
    (run_dir / "profiles" / "node_1.yaml").write_text("x: 1\n", encoding="utf-8")
    return experiment


def test_the_record_and_the_modules_file_come_first_then_the_rest(tmp_path):
    experiment = a_run(tmp_path)

    info = experiment.analysis_info("an_01")

    assert info.run_id == "an_01"
    assert [f.name for f in info.files] == [
        "analysis.yaml",
        "temporal_clustering.yaml",
        "machine.yaml",
        "resources.yaml",
    ]
    run_dir = experiment.analysis_run_dir("an_01")
    for f in info.files:
        assert f.text == (run_dir / f.name).read_text(encoding="utf-8-sig")
    assert info.files[1].text.startswith("# kept as written")


def test_the_export_holds_every_file_under_a_folder_named_for_the_run(tmp_path):
    experiment = a_run(tmp_path)
    out = tmp_path / "out" / "meta.zip"
    out.parent.mkdir()

    written = experiment.export_analysis_info("an_01", out)

    assert written == out
    info = experiment.analysis_info("an_01")
    with zipfile.ZipFile(out) as archive:
        assert archive.namelist() == [f"an_01/{f.name}" for f in info.files]
        for f in info.files:
            assert archive.read(f"an_01/{f.name}").decode("utf-8") == f.text
    assert list(out.parent.iterdir()) == [out]


def test_a_run_that_does_not_exist_is_refused(tmp_path):
    experiment = Experiment.create("meta", tmp_path)
    with pytest.raises(FileNotFoundError):
        experiment.analysis_info("an_99")
    with pytest.raises(FileNotFoundError, match="no run directory"):
        run_info.read_run_info(tmp_path / "missing", "an_99")


def test_a_run_with_nothing_to_export_says_so(tmp_path):
    empty = tmp_path / "an_05"
    empty.mkdir()
    info = run_info.read_run_info(empty, "an_05")
    assert info.files == ()
    with pytest.raises(ValueError, match="an_05 has no meta-information files"):
        run_info.export_run_info(info, tmp_path / "x.zip")
