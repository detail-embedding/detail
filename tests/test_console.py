"""A narrow stdout degrades the CLI's output; it never aborts the command.

**The class: a byte-level property of the output stream deciding whether a command runs
at all.** Python takes ``sys.stdout``'s encoding from the locale — ``cp1252`` on a
default Windows console, ASCII under ``LANG=C`` — and the default error handler *raises*,
so the first character the stream cannot represent aborts the command mid-output with a
``UnicodeEncodeError`` traceback. ``detail doctor`` printed its ``Environment`` header and
died on the check mark of its first status row. That made the one command
``.github/ISSUE_TEMPLATE/bug_report.md`` asks a reporter to paste ("Paste ``detail
doctor`` verbatim") the one command a Windows reporter could not run — and Windows is the
platform CI never exercises, so its reports are often the only signal it sends.

**How it hid.** Three ways at once. CI is Linux and the dev box is macOS, and both default
to UTF-8, so every literal encodes and no test anywhere in the suite writes to a narrow
stream. Nothing is wrong at the line where it is written: ``click.echo(f"  {glyph} …")``
is correct code, and *which* of the ~250 non-ASCII literals in ``detail/`` can reach stdout
is not a property of any one of them. And click looks like it should have caught it —
``_force_correct_text_stream`` re-wraps a stream it considers misconfigured, but
``_stream_is_misconfigured`` means *ASCII encoding only*, so a cp1252 stream is passed
through untouched and the rescue that saves ``LANG=C`` Linux does nothing for Windows.

**The rule: the CLI's output streams are configured so that no character can abort a
command.** ``detail/console.py`` registers an encode error handler that spells an
unencodable character in ASCII, and ``cli._ConsoleSafeGroup`` installs it in
``Command.main`` — the outermost frame of every invocation, before argument parsing, so
``--help`` is covered too.

**Why a transliteration and not** ``errors="replace"``. ``?`` is not always merely
degraded. It is already ``doctor``'s *unknown* glyph, so a passing check and an unknown
one would print identically — wrong information rather than missing information — and a
bound printed as ``? 2`` says the opposite of ``≥ 2``. Both are asserted below.

The sweep is over the whole command tree rather than the two commands that were reported,
because the reported symptom was not the extent of it: at the time of the fix ``detail
analysis syllable-discovery --help`` and ``detail clusterer fit --help`` were also
unreachable on Windows, and nobody had found them.
"""

import ast
import contextlib
import io
import pathlib
import sys

import click
import pytest

from detail import cli
from detail import console
from detail.api import checks as api_checks
from detail.core.dataclasses import CheckResult, CheckStatus

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The encoding of a default Windows console, and the error handler Python actually pairs
#: with it on a redirected stdout. ``surrogateescape`` rescues lone surrogates and nothing
#: else, so it raises on an unmappable character exactly as ``strict`` does.
NARROW_ENCODING = "cp1252"
NARROW_ERRORS = "surrogateescape"


def _narrow_stream() -> tuple[io.TextIOWrapper, io.BytesIO]:
    raw = io.BytesIO()
    return (
        io.TextIOWrapper(raw, encoding=NARROW_ENCODING, errors=NARROW_ERRORS),
        raw,
    )


@contextlib.contextmanager
def _narrow_console():
    """Both output streams replaced by one a default Windows console would give us."""
    stream, raw = _narrow_stream()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = stream
    try:
        yield stream, raw
    finally:
        stream.flush()
        sys.stdout, sys.stderr = real_out, real_err


def _run(args: list[str]) -> tuple[int, str, str]:
    """Exit code, decoded output, and the error handler the stream came back carrying.

    Invokes the CLI onto a narrow console the way the ``detail`` console script does.

    Through ``Command.main`` with ``standalone_mode=True``, because that is the frame the
    fix hooks and the one every real invocation enters through. A ``UnicodeEncodeError``
    is not a ``ClickException``, so it propagates out of here rather than being reported.
    """
    with _narrow_console() as (stream, raw):
        try:
            cli.main(args=args, standalone_mode=True)
            code = 0
        except SystemExit as exit_:
            code = 0 if exit_.code is None else int(exit_.code)
        installed = stream.errors
    return code, raw.getvalue().decode(NARROW_ENCODING), installed


def _help_paths() -> list[list[str]]:
    """Every command path in the tree, so a command added tomorrow is covered."""

    def walk(command, path):
        yield path
        if isinstance(command, click.Group):
            for name, sub in command.commands.items():
                yield from walk(sub, path + [name])

    return list(walk(cli.main, []))


def _fake_checks(statuses: dict[str, CheckStatus]):
    return [CheckResult(cid, status, "") for cid, status in statuses.items()]


# --------------------------------------------------------------------------- #
# The fixture bites
# --------------------------------------------------------------------------- #
def test_the_narrow_console_aborts_an_unconfigured_write():
    # Guard the guard: without this the sweep below passes vacuously on any machine,
    # because a stream that can encode everything proves nothing about one that cannot.
    stream, _ = _narrow_stream()
    with pytest.raises(UnicodeEncodeError):
        stream.write("  ✓ CUDA (GPU)\n")
        stream.flush()


def test_the_entry_point_reconfigures_the_real_stream():
    # The commands below could pass by printing nothing but ASCII. This asserts the
    # mechanism instead: the stream we installed came back carrying the handler.
    _, _, installed = _run(["--help"])
    assert installed == console.ERROR_HANDLER


# --------------------------------------------------------------------------- #
# The sweep: no command's output can abort
# --------------------------------------------------------------------------- #
def test_no_help_screen_can_abort_on_a_narrow_console():
    aborted: dict[str, str] = {}
    empty: list[str] = []
    for path in _help_paths():
        name = " ".join(["detail", *path])
        try:
            code, text, _ = _run([*path, "--help"])
        except UnicodeEncodeError as exc:
            aborted[name] = str(exc)
            continue
        if code != 0 or not text.strip():
            empty.append(f"{name} (exit={code}, {len(text)} chars)")
    assert not aborted, "help screens that abort on a cp1252 console: " + ", ".join(
        f"{k}: {v}" for k, v in aborted.items()
    )
    assert not empty, "help screens that printed nothing: " + ", ".join(empty)


def test_doctor_prints_every_row_on_a_narrow_console(monkeypatch):
    # The reported bug: `detail doctor` printed "Environment" and died on the first row.
    # Every status is represented, so every glyph in the table is exercised.
    infos = api_checks.list_checks()
    cycle = list(CheckStatus)
    statuses = {info.check_id: cycle[n % len(cycle)] for n, info in enumerate(infos)}
    monkeypatch.setattr(
        api_checks, "run_checks", lambda **kwargs: _fake_checks(statuses)
    )

    code, text, _ = _run(["doctor"])

    assert code == 1  # a `fail` is in the cycle
    missing = [info.label for info in infos if info.label not in text]
    assert not missing, "checks missing from the output: " + ", ".join(missing)


def test_a_passing_check_does_not_read_like_an_unknown_one(monkeypatch):
    # The property that rules out `errors="replace"`: it would render the check mark as
    # `?`, which is already the *unknown* glyph, so a green machine and an unmeasurable
    # one would print the same character. Plausible output instead of a crash.
    infos = api_checks.list_checks()
    ok_check, unknown_check = infos[0], infos[1]
    statuses = {info.check_id: CheckStatus.SKIPPED for info in infos}
    statuses[ok_check.check_id] = CheckStatus.OK
    statuses[unknown_check.check_id] = CheckStatus.UNKNOWN
    monkeypatch.setattr(
        api_checks, "run_checks", lambda **kwargs: _fake_checks(statuses)
    )

    _, text, _ = _run(["doctor"])

    rows = {
        info.label: line.strip().split(maxsplit=1)[0]
        for info in infos
        for line in text.splitlines()
        if line.strip().endswith(info.label)
    }
    assert rows[ok_check.label] != rows[unknown_check.label]
    assert rows[ok_check.label] != "?"


# --------------------------------------------------------------------------- #
# The class: every literal in the package, not only the ones a command prints today
# --------------------------------------------------------------------------- #
def _package_literal_characters() -> dict[str, str]:
    """Every non-ASCII character in a string literal under ``detail/``, to its file."""
    found: dict[str, str] = {}
    for path in sorted((REPO / "detail").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            for char in node.value:
                if ord(char) > 127:
                    found.setdefault(char, path.relative_to(REPO).as_posix())
    return found


def test_no_string_literal_in_the_package_can_abort_a_narrow_stream():
    # Which literal reaches stdout is not a property of the literal, so the guard covers
    # all of them rather than the ones a command happens to print today.
    console.configure_output_streams()
    characters = _package_literal_characters()
    assert len(characters) > 40, "the scan found almost nothing — it stopped scanning"

    unencodable = {}
    for char, where in characters.items():
        try:
            char.encode(NARROW_ENCODING, errors=console.ERROR_HANDLER)
        except UnicodeEncodeError as exc:
            unencodable[f"{ascii(char)} ({where})"] = str(exc)
    assert not unencodable, "literals that would abort a command: " + ", ".join(
        unencodable
    )


@pytest.mark.parametrize(
    "encoding, glyph, spelled",
    [
        # cp1252 — a Windows console. These are the ones the report was filed against.
        (NARROW_ENCODING, "✓", "+"),
        (NARROW_ENCODING, "✗", "x"),
        (NARROW_ENCODING, "≥", ">="),
        (NARROW_ENCODING, "≤", "<="),
        (NARROW_ENCODING, "→", "->"),
        (NARROW_ENCODING, "≈", "~"),
        # ASCII — a `LANG=C` Linux box. cp1252 encodes these five itself, so they never
        # reach the handler on Windows and are only reachable from the narrower stream.
        ("ascii", "—", "--"),
        ("ascii", "…", "..."),
        ("ascii", "•", "*"),
        ("ascii", "§", "S"),
        ("ascii", "±", "+/-"),
    ],
)
def test_a_meaningful_character_is_spelled_out_rather_than_blanked(
    encoding, glyph, spelled
):
    # `≥ 2` degraded to `? 2` states the opposite of the bound it replaced. These are the
    # characters where losing the identity loses the meaning, so they are named.
    console.configure_output_streams()
    encoded = glyph.encode(encoding, errors=console.ERROR_HANDLER)
    assert encoded.decode(encoding) == spelled


def test_an_unnamed_character_degrades_instead_of_raising():
    # The table is not required to be complete — the guarantee is that the command
    # finishes. An unfamiliar glyph loses its identity and nothing else.
    console.configure_output_streams()
    unnamed = "░"  # a shade block, in no CLI output today
    assert unnamed not in console._ASCII_FALLBACKS
    assert unnamed.encode(NARROW_ENCODING, errors=console.ERROR_HANDLER) == b"?"


def test_a_stream_that_cannot_be_reconfigured_is_left_alone(monkeypatch):
    # Hardening the output must never be the thing that breaks it.
    monkeypatch.setattr(sys, "stdout", object())
    monkeypatch.setattr(sys, "stderr", None)
    console.configure_output_streams()
