"""Every ``detail …`` example in the CLI guide is a command the CLI would accept.

The class: **a documented command the tool rejects before it runs.** The guide is the
page a user copies from, so an example click refuses at parse time is not a stale
sentence, it is a command that cannot work — and the reader has no way to tell our
mistake from theirs.

How it hid: nothing checks prose against the parser. ``tests/test_help.py`` and
``tests/configs/test_cli_defaults_match_models.py`` read the *decorators*, so a flag
spelled wrong in a fenced block is invisible to both, and the strict docs build only
resolves links. Four examples were wrong at once and one of them had been wrong since
the command was written: ``--reference <alias>`` (the flag takes no value; the alias
belongs to ``--video-root``), ``backfill-kinematics <emb_NN>`` (no positional exists and
three options are required), ``export attach``/``cancel`` without the required
``--experiment``, and ``add-stream --preset`` without ``--importer`` — which was a bug in
the decorator, not in the prose, and is fixed rather than documented around.

The rule: a ``detail`` line inside a fenced block in ``docs/cli.md`` parses. This runs
``parse_args`` only — no command body executes, nothing is written, and no experiment is
opened, so it costs a parse per example and belongs in the guard tier rather than beside
the CLI's behavioural tests.

Two kinds of line are skipped, both deliberately: a line carrying the elision ``…``,
which is prose standing in for arguments, and a shell construct (a pipe, ``&&``, a
command substitution), which is not a single command. Angle-bracket placeholders like
``<id>`` are substituted rather than skipped — the shape of the command is still real.
"""

import contextlib
import pathlib
import re
import shlex
import unittest.mock

import click
import pytest

from detail.cli import main

REPO = pathlib.Path(__file__).resolve().parents[1]
CLI_GUIDE = REPO / "docs" / "cli.md"

# Prose standing in for arguments, and lines that are not one command.
_ELISION = "…"
_SHELL = ("|", "&&", "$(", "`")


def _examples(text: str) -> list[str]:
    """Every ``detail …`` command line inside a fenced block, continuations joined."""
    found: list[str] = []
    for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.S):
        for line in block.replace("\\\n", " ").splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("detail "):
                found.append(line)
    return found


def _runnable(line: str) -> bool:
    return _ELISION not in line and not any(tok in line for tok in _SHELL)


def _argv(line: str) -> list[str]:
    """The example as argv, with angle-bracket placeholders made concrete."""
    return shlex.split(re.sub(r"<[^>]+>", "placeholder", line))[1:]


@contextlib.contextmanager
def _paths_need_not_exist():
    """Take ``click.Path(exists=True)`` out of the scan.

    An example names the reader's own file (``clip.mp4``, ``groups.csv``); whether it is
    on this machine says nothing about whether the command is spelled right, and a doc
    cannot make it true. The question here is the shape of the command, so the existence
    check is the one piece of validation dropped — every other type, choice and
    requirement still applies.
    """
    with unittest.mock.patch.object(
        click.types.Path, "convert", lambda self, value, param, ctx: value
    ):
        yield


def _parse(argv: list[str]) -> None:
    """Walk the group tree to the leaf command and parse its arguments.

    Raises whatever click would raise for a user who typed this, and nothing else:
    ``parse_args`` stops short of ``invoke``, so no callback runs.
    """
    command: click.Command = main
    ctx = click.Context(command, info_name="detail")
    rest = list(argv)
    while isinstance(command, click.Group):
        if not rest:
            return  # a bare group, e.g. `detail docs` — valid, nothing to parse
        name, *after = rest
        sub = command.get_command(ctx, name)
        if sub is None:
            raise click.UsageError(f"no such command: {name}")
        ctx = click.Context(sub, parent=ctx, info_name=name)
        command, rest = sub, after
    command.parse_args(ctx, rest)


def test_every_cli_guide_example_parses():
    text = CLI_GUIDE.read_text(encoding="utf-8-sig")
    lines = [line for line in _examples(text) if _runnable(line)]

    # Guard the guard: a scan that matched nothing would pass forever.
    assert len(lines) > 40, f"only {len(lines)} examples found — did the fences change?"

    failures = []
    with _paths_need_not_exist():
        for line in lines:
            try:
                _parse(_argv(line))
            except click.ClickException as e:
                failures.append(f"{line}\n    -> {e.format_message()}")
            except SystemExit:  # click exits on a usage error it formats itself
                failures.append(f"{line}\n    -> rejected by the parser")

    assert not failures, (
        "docs/cli.md shows commands the CLI refuses before running them. A reader "
        "copying one of these cannot tell our mistake from theirs:\n\n"
        + "\n".join(failures)
    )


@pytest.mark.parametrize(
    "line, why",
    [
        ("detail recording add-video --reference alias", "--reference takes no value"),
        ("detail export attach export_001", "--experiment is required"),
        ("detail nonsense --flag", "no such command"),
        ("detail recording add-stream --name x", "several required options missing"),
    ],
)
def test_the_example_scan_would_catch_a_broken_one(line, why):
    """Guard the guard, on the spellings a person actually writes."""
    with pytest.raises((click.ClickException, SystemExit)):
        _parse(_argv(line))


def test_the_extractor_reads_continuations_and_skips_prose():
    """The two skips are deliberate, and a joined continuation is one command."""
    text = (
        "```bash\n"
        "detail one --a 1 \\\n    --b 2\n"
        "detail two …  --c\n"
        "detail three --d | grep x\n"
        "detail four --e   # a trailing comment\n"
        "```\n"
    )
    found = _examples(text)

    assert found[0] == "detail one --a 1      --b 2"
    assert found[3] == "detail four --e"
    assert [line for line in found if _runnable(line)] == [found[0], found[3]]
