"""An ordinary input mistake reaches a CLI user as one ``Error:`` line, never a traceback.

**The class.** About 90 commands each wrote their own
``except (FileNotFoundError, ValueError) as e: raise click.ClickException(str(e))``. A command
that forgot, or a reader whose mistake surfaced as some other exception (a YAML
``ParserError``, a ``tarfile.ReadError``, a ``BadZipFile``), printed a Python traceback for a
missing bracket or a file picked by mistake (#33). Nothing is wrong at the line that forgets
the handler, so no behavioural test of a good input sees it.

**The rule.** A reader that meets an ordinary input mistake raises
``detail.core.errors.UserInputError`` with a message that names the thing and says what to
do. The root group (``detail/cli.py::_ConsoleSafeGroup.invoke``) turns every error in
``cli.USER_ERRORS`` into ``Error: <message>`` and exit 1, whatever command raised it. It
catches that named tuple and nothing wider: a ``KeyError``, ``TypeError`` or plain
``ValueError`` is a bug in the reader that raised it, and keeps its traceback so it can be
reported. The owner chose a dedicated type over catching ``ValueError`` at the root for
exactly that reason.

The second half of this file is the table: one row per #33 finding with a CLI step, driven
through the real ``main``, each asserting exit 1, a clean ``Error:`` line and the new
message's key words.
"""

import io
import tarfile
import zipfile

import click
import pytest
import yaml
from click.testing import CliRunner

from detail import cli
from detail.core.errors import UserInputError

#: Raised by a stub command; the root must show each as one ``Error:`` line.
NAMED = {
    "user-input": UserInputError("bad.yaml is not valid YAML (line 2, column 6): oops"),
}


class _ReaderError(UserInputError):
    """A reader's own subclass is still a user error."""


NAMED["subclass"] = _ReaderError("rig.csv row 7: expected 12 cells, found 6")

#: Raised by a stub command; the root must let each through, traceback and all.
EXCLUDED = {
    "value": ValueError("a bug that happens to be a ValueError"),
    "key": KeyError("x"),
    "type": TypeError("unsupported operand"),
    "attribute": AttributeError("'NoneType' object has no attribute 'get'"),
    "runtime": RuntimeError("plain"),
    "zero": ZeroDivisionError("division by zero"),
}


def _stub_root(group_cls: type[click.Group]) -> click.Group:
    @click.group(cls=group_cls)
    def root() -> None:
        """A copy of the root group with one command per error."""

    for name, exc in {**NAMED, **EXCLUDED}.items():

        def _raise(exc=exc) -> None:
            raise exc

        root.command(name)(_raise)
    return root


def boundary_violations(group_cls: type[click.Group]) -> list[str]:
    """What ``group_cls`` gets wrong as the root boundary, as readable strings."""
    root = _stub_root(group_cls)
    problems = []
    for name, exc in NAMED.items():
        result = CliRunner().invoke(root, [name])
        if not (
            isinstance(result.exception, SystemExit)
            and result.exit_code == 1
            and result.output == f"Error: {exc}\n"
        ):
            problems.append(f"{name}: not one Error line ({result.exception!r})")
    for name, exc in EXCLUDED.items():
        result = CliRunner().invoke(root, [name])
        if result.exception is not exc:
            problems.append(f"{name}: swallowed at the root ({result.output!r})")
    return problems


def test_the_root_group_is_the_boundary():
    assert isinstance(cli.main, cli._ConsoleSafeGroup)
    assert boundary_violations(cli._ConsoleSafeGroup) == []


def test_the_named_errors_never_widen_to_a_bug():
    for caught in cli.USER_ERRORS:
        assert not issubclass(ValueError, caught), caught
        assert not issubclass(RuntimeError, caught), caught


def test_the_root_boundary_would_catch_a_missing_handler():
    class _NoBoundary(cli._ConsoleSafeGroup):
        invoke = click.Group.invoke

    problems = boundary_violations(_NoBoundary)
    assert {p.split(":")[0] for p in problems} == set(NAMED)


def test_the_root_boundary_rejects_a_broad_catch(monkeypatch):
    monkeypatch.setattr(cli, "USER_ERRORS", (Exception,))
    problems = boundary_violations(cli._ConsoleSafeGroup)
    assert {p.split(":")[0] for p in problems} == set(EXCLUDED)


# --------------------------------------------------------------------------- #
# One row per finding with a CLI step (#33 section A).
# --------------------------------------------------------------------------- #
def _invoke(*args: str):
    return CliRunner().invoke(cli.main, [str(a) for a in args])


def _clean_error(result, *words: str) -> str:
    """The ``Error:`` line of a run that failed cleanly, checked for ``words``."""
    assert isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    errors = [l for l in result.output.splitlines() if l.startswith("Error: ")]
    assert len(errors) == 1, result.output
    line = errors[0]
    for word in words:
        assert word in result.output, (word, result.output)
    return line


def _bundle(tmp_path, text: str):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.yaml").write_text(text, encoding="utf-8")
    return bundle


def test_finding_1_a_manifest_with_a_yaml_mistake(experiment_dir, tmp_path):
    bundle = _bundle(tmp_path, "recordings:\n  - id: m1\n   video: a.mp4\n")
    result = _invoke(
        "recording", "import", "--experiment", experiment_dir, "--bundle", bundle
    )
    _clean_error(result, "manifest.yaml is not valid YAML (line 3, column 4)")


def test_finding_1_recordings_as_plain_names(experiment_dir, tmp_path):
    bundle = _bundle(tmp_path, "recordings: [m1, m2]\n")
    result = _invoke(
        "recording", "import", "--experiment", experiment_dir, "--bundle", bundle
    )
    _clean_error(
        result,
        "recording[0]: each entry under recordings must be a mapping with an id, not 'm1'",
    )


@pytest.fixture
def broken_yaml(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("names: [a, b\nedges: []\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "command",
    [
        ("skeleton", "import"),
        ("skeleton", "create", "sk", "--from"),
        ("keypoints", "examples", "import", "--experiment", "{exp}"),
        ("experiment", "import-heading", "--experiment", "{exp}"),
        ("experiment", "import-stream-preset", "--experiment", "{exp}"),
    ],
    ids=lambda c: " ".join(c[:2]),
)
def test_finding_2_a_yaml_importer_names_the_file_and_line(
    command, experiment_dir, broken_yaml
):
    args = [str(experiment_dir) if a == "{exp}" else a for a in command]
    _clean_error(
        _invoke(*args, broken_yaml), "broken.yaml is not valid YAML (line 2, column 6)"
    )


def test_finding_2_a_binary_file_is_not_a_text_file(tmp_path):
    png = tmp_path / "some.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe")
    line = _clean_error(_invoke("skeleton", "import", png))
    assert line == f"Error: {png} is not a text file"


def test_finding_3_a_damaged_recording_yaml(experiment_dir, recording_dir):
    path = recording_dir / "recording.yaml"
    path.write_text("schema_version: 1\nid: recording_001\nname: [m1\n", "utf-8")
    status = _invoke("experiment", "status", "--experiment", experiment_dir)
    assert status.exit_code == 0, status.output
    assert "recordings: 1" in status.output
    assert "recording_001  (damaged: " in status.output
    assert "recording.yaml is not valid YAML (line 4" in status.output
    create = _invoke(
        "recording", "create", "--experiment", experiment_dir, "--name", "m2"
    )
    _clean_error(create, f"{path} is not valid YAML (line 4")


def test_finding_3_a_damaged_config_yaml(experiment_dir):
    config = experiment_dir / "config.yaml"
    config.write_text(config.read_text("utf-8") + "notes: [x\n", "utf-8")
    result = _invoke("experiment", "status", "--experiment", experiment_dir)
    _clean_error(result, "config.yaml is not valid YAML (line")


@pytest.mark.parametrize(
    "text, words",
    [
        (
            "schema_version: 1\nlabels:\n  recording_001: l.csv\n   bad: x\n",
            "labels.yaml is not valid YAML (line 4, column 7)",
        ),
        (
            "schema_version: 1\nlabels:\n  recording_001: [l.csv]\n",
            "labels['recording_001']: the file must be one filename, not a list",
        ),
        (
            "schema_version: 1\nlabels:\n"
            "  recording_001: {file: l.csv, covered_behaviors: groom}\n",
            "covered_behaviors must be a list of behavior names",
        ),
    ],
    ids=["yaml", "file-list", "covered-string"],
)
def test_finding_4_import_labels_manifest(
    text, words, experiment_dir, recording_dir, tmp_path
):
    (tmp_path / "l.csv").write_text("start_frame,end_frame,label\n0,1,groom\n", "utf-8")
    manifest = tmp_path / "labels.yaml"
    manifest.write_text(text, encoding="utf-8")
    result = _invoke(
        "recording",
        "import-labels",
        "--experiment",
        experiment_dir,
        "--manifest",
        manifest,
    )
    _clean_error(result, words)


def _tar_with_manifest(path, manifest: dict) -> None:
    body = yaml.safe_dump(manifest).encode()
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo("export.yaml")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))


@pytest.mark.parametrize(
    "make, why",
    [
        (lambda p: p.write_text("hello\n", "utf-8"), "not a tar archive"),
        (lambda p: p.write_bytes(b""), "the file is empty"),
        (lambda p: zipfile.ZipFile(p, "w").close(), "not a tar archive"),
        (
            lambda p: _tar_with_manifest(p, {"id": "e", "schema_version": "one"}),
            "its manifest's schema_version is 'one', not a number",
        ),
        (
            lambda p: _tar_with_manifest(p, {"id": 5, "schema_version": 1}),
            "its manifest's id is 5, not a name",
        ),
    ],
    ids=["text", "empty", "zip", "text-version", "numeric-id"],
)
def test_finding_5_import_of_something_that_is_not_an_export(make, why, tmp_path):
    archive = tmp_path / "thing.tar"
    make(archive)
    line = _clean_error(_invoke("experiment", "import", archive))
    assert line == (
        f"Error: thing.tar is not a DETAIL export, or it is incomplete ({why}); "
        "copy it again, or export the experiment again"
    )


def test_finding_5_an_archive_cut_inside_the_manifest(tmp_path):
    whole = tmp_path / "whole.tar"
    _tar_with_manifest(whole, {"id": "e", "schema_version": 1, "pad": "x" * 2000})
    cut = tmp_path / "cut.tar"
    cut.write_bytes(whole.read_bytes()[:1200])
    _clean_error(_invoke("experiment", "import", cut), "cut.tar is not a DETAIL export")


@pytest.fixture
def model_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("DETAIL_MODEL_DIR", str(tmp_path / "models"))
    return tmp_path / "models" / "imported"


def test_finding_6_restore_of_a_checkpoint_not_a_zip(model_catalog, tmp_path):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"not a zip")
    _clean_error(
        _invoke("model", "restore", pt),
        "best.pt is not a model export",
        "detail model import",
    )


def test_finding_6_restore_of_an_empty_model_yaml_leaves_nothing(
    model_catalog, tmp_path
):
    archive = tmp_path / "m.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("mymodel/model.yaml", "")
    _clean_error(
        _invoke("model", "restore", archive),
        "m.zip: mymodel/model.yaml is not a model record",
    )
    assert list(model_catalog.iterdir()) == []


_SPATIAL_HEADER = (
    "kind,name,active_start,active_end,frame_idx,vertex_index,x,y,distance_mm,"
    "pixels_per_mm,ref_width,ref_height\n"
)


@pytest.mark.parametrize(
    "command, body, words",
    [
        (
            "import-spatial",
            "roi,z,,,0,0,0.1,0.1,,,,\nroi,z,,,0,1,0.5\n",
            "rig.csv row 3: expected 12 cells, found 7 (no 'y')",
        ),
        (
            "import-calibration",
            "calibration,cal,,,,,,,,2.0,640,0\n",
            "ref_width and ref_height must be positive",
        ),
    ],
    ids=["short-row", "zero-size"],
)
def test_finding_8_spatial_csv(
    command, body, words, experiment_dir, recording_dir, tmp_path
):
    csv_path = tmp_path / "rig.csv"
    csv_path.write_text(_SPATIAL_HEADER + body, encoding="utf-8")
    result = _invoke(
        "recording",
        command,
        "--experiment",
        experiment_dir,
        "--recording",
        "recording_001",
        "--csv",
        csv_path,
    )
    _clean_error(result, words)


@pytest.mark.parametrize(
    "content, words",
    [
        (b"", "missing columns"),
        (b"a,b\n1,2\n", "missing columns"),
        (b"\x89PNG\r\n\x1a\n\xff", "is not UTF-8 text"),
    ],
    ids=["empty", "other-columns", "binary"],
)
def test_finding_9_query_from_a_csv_that_is_not_a_clip_list(
    content, words, experiment_dir, recording_dir, tmp_path
):
    path = tmp_path / "clips.csv"
    path.write_bytes(content)
    result = _invoke(
        "analysis",
        "query",
        "--experiment",
        experiment_dir,
        "--search",
        "recording_001",
        "--embed-run",
        "emb_01",
        "--from-csv",
        path,
    )
    _clean_error(result, words)


def test_finding_9_query_eval_checks_out_before_evaluating(
    experiment_dir, recording_dir, tmp_path
):
    """No embedding run exists, so an evaluation would fail on that; failing on ``--out``
    instead shows the check ran first."""
    out = tmp_path / "notes.csv"
    out.write_text("a,b\n1,2\n", encoding="utf-8")
    result = _invoke(
        "analysis",
        "query-eval",
        "--experiment",
        experiment_dir,
        "--search",
        "recording_001",
        "--embed-run",
        "emb_01",
        "--behavior",
        "groom",
        "--out",
        out,
    )
    _clean_error(result, f"--out {out} is not a query-eval metrics CSV")
    assert out.read_text("utf-8") == "a,b\n1,2\n"
