"""Every CLI path option expands ``~``, because every one uses ``cli_paths.UserPath``.

**The class.** A shell expands ``~`` only unquoted and at the start of a word, so
``--dir "~/Documents/DetailExperiments"`` (the form the docs show), a path pasted from a
script, or a shell that leaves ``~`` alone reaches the command as a literal ``~``.
``click.Path`` keeps it: ``experiment create`` printed ``created experiment: ~/Documents/…``
and made a folder named ``~`` in the current directory, which the GUI never shows (#31,
finding 21). Nothing is wrong at the option that declares it, and every test passes an
absolute ``tmp_path``.

**The rule.** A path option or argument is typed ``UserPath`` (``detail/cli_paths.py``),
``click.Path`` with ``~`` expanded first. This scan fails on a ``click.Path(...)`` call
anywhere else in ``detail/``, so a new option that uses it directly fails here, and the
behaviour test runs ``experiment create`` with a quoted ``~``.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap

from click.testing import CliRunner

import detail
from detail import cli

_PACKAGE = pathlib.Path(detail.__file__).parent
_SEAM = "cli_paths.py"


def _bare_click_paths(source: str) -> int:
    """How many calls in ``source`` build a ``click.Path`` (as ``click.Path(...)``, or a
    bare ``Path(...)`` imported from click), read from the parse tree."""
    tree = ast.parse(source)
    from_click = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "click"
        for alias in node.names
        if alias.name == "Path"
    }
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "Path"
            and isinstance(func.value, ast.Name)
            and func.value.id == "click"
        ):
            count += 1
        elif isinstance(func, ast.Name) and func.id in from_click:
            count += 1
    return count


def test_no_path_option_uses_click_path_directly():
    found = {}
    for py in sorted(_PACKAGE.rglob("*.py")):
        rel = py.relative_to(_PACKAGE).as_posix()
        if rel == _SEAM:
            continue
        try:
            source = py.read_text(encoding="utf-8-sig")
        except FileNotFoundError:  # a module another test planted and removed mid-run
            continue
        n = _bare_click_paths(source)
        if n:
            found[rel] = n
    assert found == {}, f"use detail.cli_paths.UserPath, not click.Path: {found}"


def test_the_scan_would_catch_a_click_path():
    source = textwrap.dedent("""
        import click
        from click import Path as P

        a = click.option("--dir", type=click.Path(
            file_okay=False))
        b = click.argument("p", type=P(exists=True))
        c = '''type=click.Path( in a docstring'''
        d = UserPath(exists=True)
        """)
    assert _bare_click_paths(source) == 2


def test_a_quoted_tilde_lands_in_the_home_folder(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli.main, ["experiment", "create", "study", "--dir", "~/exps"]
    )
    assert result.exit_code == 0, result.output
    assert (home / "exps" / "study" / "config.yaml").is_file()
    assert not (tmp_path / "~").exists()
    assert result.output.strip() == f"created experiment: {home / 'exps' / 'study'}"


def test_an_existing_file_is_checked_after_expanding(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    (home / "s.yaml").write_text("names: [nose, tail]\n", encoding="utf-8")
    result = CliRunner().invoke(cli.main, ["skeleton", "import", "~/s.yaml"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("imported skeleton: ")
