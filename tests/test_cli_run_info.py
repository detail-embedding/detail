"""``analysis info``: the CLI's half of the run page's *Meta-information…* window."""

from __future__ import annotations

import zipfile

from click.testing import CliRunner

from detail import cli
from tests.api.test_run_info import MODULE_TEXT, a_run


def _invoke(*args):
    return CliRunner().invoke(cli.main, ["analysis", "info", *args])


def test_info_prints_every_file_in_the_windows_order(tmp_path):
    experiment = a_run(tmp_path)

    result = _invoke("an_01", "--experiment", str(experiment.dir))

    assert result.exit_code == 0, result.output
    headers = [line for line in result.output.splitlines() if line.startswith("# --- ")]
    assert headers == [
        "# --- analysis.yaml ---",
        "# --- temporal_clustering.yaml ---",
        "# --- machine.yaml ---",
        "# --- resources.yaml ---",
    ]
    assert MODULE_TEXT.rstrip("\n") in result.output


def test_info_export_writes_the_zip(tmp_path):
    experiment = a_run(tmp_path)
    out = tmp_path / "meta.zip"

    result = _invoke("an_01", "--experiment", str(experiment.dir), "--export", str(out))

    assert result.exit_code == 0, result.output
    assert f"wrote {out}" in result.output
    with zipfile.ZipFile(out) as archive:
        assert "an_01/temporal_clustering.yaml" in archive.namelist()


def test_info_on_a_run_that_does_not_exist_fails_by_name(tmp_path):
    experiment = a_run(tmp_path)

    result = _invoke("an_99", "--experiment", str(experiment.dir))

    assert result.exit_code != 0
    assert "no analysis run 'an_99'" in result.output
