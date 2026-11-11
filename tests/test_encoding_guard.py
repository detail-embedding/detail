"""Text I/O names its encoding; it never takes the one the OS happens to be set to.

**The class: a value read from the environment deciding how bytes become text.**
``open(p)`` / ``read_text()`` / ``write_text()`` with no ``encoding=`` use
``locale.getpreferredencoding(False)`` — UTF-8 on the Linux CI box and on macOS, and
**cp1252 on Windows**. Nothing is wrong at the line where it is written, and the same line
does two different things on two machines.

**What it costs on Windows, before this was fixed.** cp1252 leaves 0x81,
0x8D, 0x8F, 0x90 and 0x9D undefined, and UTF-8 continuation bytes land on them constantly
(``←`` U+2190 is ``E2 86 90``), so seven files in ``detail/`` could not be *read* at all
and `analysis comparison` could not *write* its ``summary.md`` (the ``★`` in
``cluster_eval_report._write_summary``). The quieter half is worse than the crash: a name
inside cp1252 is written as cp1252 bytes into a ``recording.yaml`` that everything else
reads as UTF-8, so the corruption is silent, durable, and in the user's data.

**How it hid.** CI is Linux and the dev box is macOS. Both are UTF-8, so every one of these
call sites is correct there, and no behavioural test can tell a file that names its encoding
from one that inherits a matching default.

**The rule: reads take ``utf-8-sig``, writes take ``utf-8``.** One rule rather than a
judgment per call site. On a read, tolerating a byte-order mark costs nothing and is
identical to ``utf-8`` when there is none — and on Windows a BOM arrives from editors and
shells we do not control (PowerShell's ``Out-File`` writes one), so a read that chokes on
one is a bug we have already hit. On a write we must never emit one, because every other
reader of our files would then see it as content.

This scan is the widened form of one that already existed for ``detail/importers/`` alone,
written because "an importer reads files the user hands it". That was true and too narrow:
`storage/` writes the user's experiment, and the same locale codec decides what lands in it.
The finding was that the rule was scoped to importers when it belonged to the package.

``subprocess`` is covered too, one layer out: ``text=True`` is ``encoding=<locale>``, and a
non-Python child does not write the locale (see
``test_every_subprocess_decode_names_its_encoding``).

Not covered here: ``av.open`` / ``os.open`` / ``tarfile.open`` / ``Experiment.open`` are not
text file I/O, and a binary mode has no codec.
"""

import ast
import pathlib

import pytest

import detail

PACKAGE = pathlib.Path(detail.__file__).parent
REPO = PACKAGE.parent

#: Every tree the rule covers. ``tests/`` is here for its own sake, not for tidiness: the
#: source-scanning guards read ``detail/`` and ``DESIGN_PLAN.md`` back, so on Windows an
#: unencoded read in a test made ``pytest -m guard`` unrunnable — 14 guard files failing
#: and 2 erroring at collection, every one of the 52 error lines a ``UnicodeDecodeError``.
#: A gate that cannot run on a platform is not a gate on that platform.
#: ``scripts/`` is here because the rule was scoped to the trees somebody remembered, and the
#: one it left out is the one that rewrites the user's experiment: the layout migration reads
#: and rewrites every ``*.yaml`` under the workspace, and ``UnicodeDecodeError`` is a
#: ``ValueError``, so its ``except (OSError, yaml.YAMLError)`` does not catch it — the
#: migration aborts *after* step 1 has already renamed ``sessions/`` to ``recordings/``.
ROOTS = ("detail", "tests", "scripts")

#: Modules whose own ``read_text`` / ``write_text`` *is* the encoding-naming wrapper, so a
#: call through them is the fix rather than the bug. Kept to this one entry deliberately: a
#: second seam means two answers to "how does this package read a file".
TEXT_IO_SEAMS = frozenset({"atomic"})

#: ``.open()`` on one of these is not opening a text file.
NOT_FILE_OPEN = frozenset(
    {
        "av",
        "tarfile",
        "os",
        "zipfile",
        "gzip",
        "Experiment",
        "zarr",
        "cv2",
        "Image",
        "webbrowser",
        # A `QDialog`'s own `open()` — show it modally without a nested event loop
        # (`gui/full_frame.py`). Named for the variable rather than a class because that is
        # what the receiver is at the call site; a `Path` called `dialog` would slip past, and
        # there is none in this repo.
        "dialog",
    }
)

#: The encoding each direction must name, and why the two differ (see the module docstring).
READ_ENCODING = "utf-8-sig"
WRITE_ENCODING = "utf-8"

_WRITE_MODE_CHARS = frozenset("wax+")


def _open_mode(call: ast.Call, *, method: bool) -> str | None:
    """An ``open`` call's literal mode, or ``None`` when it is computed.

    The mode's position depends on which ``open`` this is: ``open(path, mode)`` puts it
    second, ``path.open(mode)`` puts it *first*, because the path is the receiver. Reading
    ``args[1]`` for both made every ``path.open("w")`` look like a read — so the guard asked
    those writers for ``utf-8-sig`` and seven CSV and log writers in driver scripts
    duly emit a byte-order mark, which this file's own rule exists to forbid.
    """
    for keyword in call.keywords:
        if keyword.arg == "mode":
            value = keyword.value
            return value.value if isinstance(value, ast.Constant) else None
    index = 0 if method else 1
    if len(call.args) > index:
        arg = call.args[index]
        return arg.value if isinstance(arg, ast.Constant) else None
    return "r"


def text_io_calls(tree: ast.AST):
    """``(call, direction)`` for every text read/write in a parsed module."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name, receiver = func.id, ""
        elif isinstance(func, ast.Attribute):
            name, receiver = func.attr, ast.unparse(func.value)
        else:
            continue
        if name in ("read_text", "write_text"):
            if receiver.split(".")[-1] in TEXT_IO_SEAMS:
                continue  # our own wrapper — it names the encoding, that is its job
            yield node, "read" if name == "read_text" else "write"
        elif name == "open":
            if receiver.split(".")[0] in NOT_FILE_OPEN:
                continue
            mode = _open_mode(node, method=isinstance(func, ast.Attribute))
            if mode is None or "b" in mode:
                continue  # computed mode, or binary — neither touches a codec
            yield node, "write" if set(mode) & _WRITE_MODE_CHARS else "read"


#: A bare ``.read_text`` / ``.write_text`` that is legitimately not a call. Claims, not
#: exemptions — each says why, and ``test_every_uncalled_allowlist_entry_still_matches`` fails on one
#: that no longer matches.
UNCALLED_ALLOWED = {
    "tests/storage/test_recording.py": "captures the unbound Path.read_text to forward to "
    "from a counting monkeypatch; it passes *args through, so the encoding survives",
}


def uncalled_text_io(tree: ast.AST):
    """``(node, name)`` for every ``.read_text`` / ``.write_text`` named but not called here.

    The shape that got past the call-based scan above, and the reason it matters: handing the
    *bound method* to a helper that invokes it — ``atomic.retry_on_lock(path.read_text)`` —
    reaches ``Path.read_text()`` with no arguments, so there is no call site left where an
    ``encoding=`` could be written. Fourteen of these survived the codemod that fixed 124 call
    sites, including ``read_recording``, the canonical reader of the user's ``recording.yaml``:
    the file was written UTF-8 thirty lines below and read back on the locale codec.
    ``core/atomic.py::read_text`` is the seam they use now.
    """
    called = {
        id(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr in ("read_text", "write_text") and id(node) not in called:
            yield node, node.attr


def _declared_encoding(call: ast.Call) -> str | None | type:
    """The literal ``encoding=`` argument, ``None`` if absent, ``NotImplemented`` if computed."""
    for keyword in call.keywords:
        if keyword.arg == "encoding":
            value = keyword.value
            return value.value if isinstance(value, ast.Constant) else NotImplemented
    return None


def _offenders(root: pathlib.Path) -> list[str]:
    out = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding=READ_ENCODING)
        lines = source.splitlines()
        for call, direction in text_io_calls(ast.parse(source)):
            declared = _declared_encoding(call)
            wanted = READ_ENCODING if direction == "read" else WRITE_ENCODING
            if declared == wanted or declared is NotImplemented:
                continue
            where = f"{path.relative_to(REPO).as_posix()}:{call.lineno}"
            saw = "no encoding=" if declared is None else f'encoding="{declared}"'
            out.append(
                f"{where} ({direction}, {saw}): {lines[call.lineno - 1].strip()}"
            )
    return out


@pytest.mark.parametrize("root", ROOTS)
def test_every_text_io_call_names_its_encoding(root):
    offenders = _offenders(REPO / root)
    assert not offenders, (
        f"text I/O must name its encoding — {READ_ENCODING!r} to read, "
        f"{WRITE_ENCODING!r} to write — so a file means the same on Windows as on "
        "Linux/macOS:\n" + "\n".join(offenders)
    )


def test_the_scan_would_catch_one():
    # Guard the guard: a scan that matches nothing today passes vacuously forever.
    caught = _offenders_in_source(
        "\n".join(
            [
                "from pathlib import Path",
                "def f(p):",
                "    return Path(p).read_text()",
            ]
        )
    )
    assert len(caught) == 1 and caught[0][1] == "read"


def _uncalled_offenders(root: pathlib.Path) -> list[str]:
    out = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel in UNCALLED_ALLOWED:
            continue
        source = path.read_text(encoding=READ_ENCODING)
        lines = source.splitlines()
        for node, name in uncalled_text_io(ast.parse(source)):
            out.append(
                f"{rel}:{node.lineno} ({name}): {lines[node.lineno - 1].strip()}"
            )
    return out


@pytest.mark.parametrize("root", ROOTS)
def test_no_text_io_method_is_handed_off_uncalled(root):
    offenders = _uncalled_offenders(REPO / root)
    assert not offenders, (
        "a bare .read_text / .write_text passed to a helper is invoked with no arguments, "
        "so it takes the locale codec and there is no call site left to name an encoding — "
        "use core/atomic.py::read_text (or a lambda that names it):\n"
        + "\n".join(offenders)
    )


def test_the_uncalled_scan_would_catch_one():
    # Guard the guard, in the exact spelling that survived the codemod.
    caught = list(
        uncalled_text_io(
            ast.parse(
                "atomic.retry_on_lock(path.read_text)\nPath(p).read_text('utf-8')"
            )
        )
    )
    assert [name for _, name in caught] == ["read_text"]
    assert caught[0][0].lineno == 1  # the called one on line 2 is not an offender


def test_every_uncalled_allowlist_entry_still_matches():
    # An entry that excuses nothing is a claim that has gone stale, not a free pass.
    for rel, reason in UNCALLED_ALLOWED.items():
        path = REPO / rel
        assert path.is_file(), f"{rel} is gone; drop its entry ({reason})"
        found = list(
            uncalled_text_io(ast.parse(path.read_text(encoding=READ_ENCODING)))
        )
        assert (
            found
        ), f"{rel} no longer hands off an uncalled text-I/O method; drop its entry"


def test_the_scan_reads_the_direction_off_the_mode():
    # A wrong direction would demand a BOM-emitting write or a BOM-intolerant read.
    directions = [
        d
        for _, d in text_io_calls(
            ast.parse(
                "\n".join(
                    [
                        "open(p)",
                        "open(p, 'r')",
                        "open(p, 'w')",
                        "open(p, 'a')",
                        "open(p, mode='w')",
                        # The method form puts the mode first — the path is the receiver.
                        # Reading it as `args[1]` called every one of these a read, which is
                        # how seven writers ended up asking for a byte-order mark.
                        "p.open()",
                        "p.open('r')",
                        "p.open('w')",
                        "p.open('w', newline='')",
                        "p.open('a')",
                        "p.open(mode='w')",
                    ]
                )
            )
        )
    ]
    assert directions == ["read", "read", "write", "write", "write"] + [
        "read",
        "read",
        "write",
        "write",
        "write",
        "write",
    ]


def test_the_scan_skips_what_has_no_codec():
    # A false positive here would put `encoding=` on a call that raises TypeError, so the
    # exclusions matter as much as the hits.
    skipped = "\n".join(
        [
            "av.open(p)",
            "os.open(p, flags)",
            "tarfile.open(p)",
            "Experiment.open(d)",
            "open(p, 'rb')",
            "open(p, mode)",
        ]
    )
    assert not list(text_io_calls(ast.parse(skipped)))


#: The ``subprocess`` entry points that can decode a child's output.
_SUBPROCESS_CALLS = frozenset(
    {"run", "Popen", "check_output", "check_call", "call", "getoutput"}
)


def _is_true(call: ast.Call, name: str) -> bool:
    return any(
        k.arg == name and isinstance(k.value, ast.Constant) and k.value.value is True
        for k in call.keywords
    )


def _child_is_this_interpreter(call: ast.Call) -> bool:
    """A literal argv starting with ``sys.executable``.

    Exempt because the two sides of the pipe agree by construction: a Python child writes a
    pipe on the same locale codec a bare ``text=True`` decodes with (and under
    ``PYTHONUTF8=1`` both switch together). Every other program writes what *it* writes —
    git UTF-8, PowerShell the console codepage — and the locale is not a guess at that.
    """
    if not call.args or not isinstance(call.args[0], ast.List) or not call.args[0].elts:
        return False
    return ast.unparse(call.args[0].elts[0]) == "sys.executable"


#: Calls whose child is this interpreter through a variable the AST cannot see into. Claims:
#: ``test_every_subprocess_allowlist_entry_still_matches`` fails on one that excuses nothing.
SUBPROCESS_ALLOWED = {
}


def undeclared_subprocess_decodes(tree: ast.AST):
    """Every ``subprocess`` call that decodes a non-Python child's output on the locale."""
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _SUBPROCESS_CALLS
            and ast.unparse(node.func.value) == "subprocess"
        ):
            continue
        if not (_is_true(node, "text") or _is_true(node, "universal_newlines")):
            continue
        if _declared_encoding(node) is not None or _child_is_this_interpreter(node):
            continue
        yield node


@pytest.mark.parametrize("root", ROOTS)
def test_every_subprocess_decode_names_its_encoding(root):
    """The same class, one layer out: ``text=True`` is ``encoding=<locale>``.

    Found on Windows (PR #436): ``git log`` writes UTF-8, and a curly quote's ``9D`` byte is
    undefined in cp1252. ``subprocess`` decodes on a reader thread, which *swallows* the
    ``UnicodeDecodeError`` — ``returncode`` 0, ``stdout`` **None**, and a test comparing commit
    bodies failed with a ``TypeError`` far from the cause. The same decode in
    ``executor/local.py::_is_our_worker`` read PowerShell's console-codepage output, so a live
    worker whose run dir held an ``é`` was reported not ours — the direction that lets a
    re-submit stack a second worker onto its device.
    """
    out = []
    for path in sorted((REPO / root).rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel in SUBPROCESS_ALLOWED:
            continue
        source = path.read_text(encoding=READ_ENCODING)
        lines = source.splitlines()
        for call in undeclared_subprocess_decodes(ast.parse(source)):
            out.append(f"{rel}:{call.lineno}: {lines[call.lineno - 1].strip()}")
    assert not out, (
        "a subprocess decoded with text=True and no encoding= takes the locale codec, which "
        "is not what git / PowerShell / nvidia-smi write — name it (and errors=, if a bad "
        "byte must not turn stdout into None):\n" + "\n".join(out)
    )


def test_the_subprocess_scan_would_catch_one():
    # Guard the guard, and its exemption: a Python child is fine, every other one is not.
    source = "\n".join(
        [
            "subprocess.run(['git', 'log'], capture_output=True, text=True)",
            "subprocess.Popen(cmd, stdout=PIPE, universal_newlines=True)",
            "subprocess.run(['git', 'log'], capture_output=True, encoding='utf-8')",
            "subprocess.run([sys.executable, '-c', 'x'], capture_output=True, text=True)",
            "subprocess.run(['git', 'log'], capture_output=True)",
        ]
    )
    caught = [c.lineno for c in undeclared_subprocess_decodes(ast.parse(source))]
    assert caught == [1, 2]


def test_every_subprocess_allowlist_entry_still_matches():
    for rel, reason in SUBPROCESS_ALLOWED.items():
        path = REPO / rel
        assert path.is_file(), f"{rel} is gone; drop its entry ({reason})"
        tree = ast.parse(path.read_text(encoding=READ_ENCODING))
        assert list(
            undeclared_subprocess_decodes(tree)
        ), f"{rel} no longer decodes on the locale; drop its entry"


def _offenders_in_source(source: str) -> list[tuple[int, str]]:
    return [
        (call.lineno, direction)
        for call, direction in text_io_calls(ast.parse(source))
        if _declared_encoding(call) is None
    ]
