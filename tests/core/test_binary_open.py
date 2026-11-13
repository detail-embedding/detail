"""Every ``os.open`` names ``O_BINARY``; a file's bytes survive being flushed on Windows.

**The class: a low-level open in the C runtime's text mode.** On Windows ``os.open`` goes
through the MSVC CRT, and without ``os.O_BINARY`` the descriptor is in *text mode*. Two
things then happen silently: opening a file read/write cuts a trailing Ctrl-Z (0x1A) off it,
and ``os.write`` turns each ``\\n`` into ``\\r\\n`` (``os.read`` does the reverse, and stops at a
Ctrl-Z). Python's own ``open()`` and ``tempfile.mkstemp`` are not affected — both open binary at
the CRT level — which is why only ``os.open`` is scanned here.

**What it cost.** ``atomic._flush_to_disk`` opened each file ``O_RDWR`` to ``fsync`` it, and
``flush_tree`` runs it over every file of a run as the run completes. Any output file whose
last byte was 0x1A lost it: about one compressed zarr chunk in 256. A detection run on the
multi-animal guide's demo finished ``complete``, and every tracking run on it then failed with
``Zstd decompression error: invalid input data`` on the same two chunks, each exactly one byte
short of what the worker had written and read back. The damage is on disk, under a record that
says the run is fine.

**How it hid.** CI is Linux and the dev box is macOS, where ``os.O_BINARY`` does not exist and
there is no text mode. On Windows the loss is data-dependent — a run whose chunks all happen to
end in another byte is untouched — so most runs were fine.

**The rule: every ``os.open`` names ``O_BINARY``** (``atomic.O_BINARY``, which is ``0`` off
Windows), read-only and POSIX-only opens included. One rule rather than a judgment per call: a
lock file that is never written is harmless in text mode today, and the next person to write to
one should not have to know that.
"""

import ast
import os
import pathlib

import pytest

import detail
from detail.core import atomic

REPO = pathlib.Path(detail.__file__).parent.parent

#: Every tree whose code runs on a user's machine or the developer's.
ROOTS = ("detail", "scripts", "research", "tests")


def _names_o_binary(flags: ast.expr) -> bool:
    for node in ast.walk(flags):
        if isinstance(node, ast.Name) and node.id == "O_BINARY":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "O_BINARY":
            return True
        if isinstance(node, ast.Constant) and node.value == "O_BINARY":
            return True  # getattr(os, "O_BINARY", 0), where detail is not importable
    return False


def text_mode_opens(source: str) -> list[int]:
    """Line numbers of every ``os.open`` whose flags do not name ``O_BINARY``.

    Fail-closed: flags held in a variable are reported too, because the scan cannot see what
    the variable holds. ``from os import open`` is reported wherever it appears.
    """
    bad = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            bad += [node.lineno for alias in node.names if alias.name == "open"]
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "open"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        ):
            continue
        flags = node.args[1] if len(node.args) > 1 else None
        for keyword in node.keywords:
            if keyword.arg == "flags":
                flags = keyword.value
        if flags is None or not _names_o_binary(flags):
            bad.append(node.lineno)
    return bad


def _sources():
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            if "views" in path.relative_to(REPO).parts:
                continue
            yield path


def test_every_os_open_names_o_binary():
    offenders = [
        f"{path.relative_to(REPO).as_posix()}:{line}"
        for path in _sources()
        for line in text_mode_opens(path.read_text(encoding="utf-8-sig"))
    ]
    assert not offenders, (
        "os.open without O_BINARY opens a text-mode descriptor on Windows, which cuts a "
        "trailing Ctrl-Z off a file opened read/write and rewrites newlines. Add "
        "`| atomic.O_BINARY` to the flags:\n  " + "\n  ".join(offenders)
    )


def test_scan_is_not_vacuous():
    """The package does call ``os.open``, so the scan has something to read."""
    calls = sum(
        "os.open(" in path.read_text(encoding="utf-8-sig")
        for path in _sources()
        if path.name != pathlib.Path(__file__).name
    )
    assert calls >= 4


@pytest.mark.parametrize(
    "source",
    [
        "fd = os.open(path, os.O_RDWR)",
        "fd = os.open(path, os.O_RDONLY)",
        "fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)",
        "fd = os.open(path, flags=os.O_WRONLY | os.O_APPEND)",
        "fd = os.open(path, flags)",
        "fd = os.open(\n    path,\n    os.O_RDWR\n    | os.O_CREAT,\n)",
        "from os import open as raw_open",
    ],
)
def test_scan_catches_a_text_mode_open(source):
    assert text_mode_opens(source)


@pytest.mark.parametrize(
    "source",
    [
        "fd = os.open(path, os.O_RDWR | O_BINARY)",
        "fd = os.open(path, os.O_RDWR | atomic.O_BINARY, 0o644)",
        'fd = os.open(path, os.O_WRONLY | getattr(os, "O_BINARY", 0))',
        "fd = os.open(path, flags=os.O_RDONLY | os.O_BINARY)",
        "with path.open('rb') as f: pass",
    ],
)
def test_scan_accepts_a_binary_open(source):
    assert not text_mode_opens(source)


@pytest.mark.parametrize(
    "content",
    [b"data\x1a", b"\x28\xb5\x2f\xfd" + bytes(range(256)) + b"\x1a", b"a\r\n\x1a"],
    ids=["short", "zstd-frame", "crlf"],
)
def test_flush_tree_keeps_a_trailing_ctrl_z(tmp_path, content):
    """The behaviour, on the platform that has a text mode. Trivially true off Windows."""
    chunk = tmp_path / "part_001.zarr" / "bbox" / "c" / "0" / "0"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(content)

    atomic.flush_tree(tmp_path)

    assert chunk.read_bytes() == content


def test_replace_keeps_a_trailing_ctrl_z(tmp_path):
    src, dst = tmp_path / "chunk.tmp", tmp_path / "chunk"
    src.write_bytes(b"payload\x1a")

    atomic.replace(src, dst)

    assert dst.read_bytes() == b"payload\x1a"


def test_o_binary_is_the_platform_flag():
    assert atomic.O_BINARY == getattr(os, "O_BINARY", 0)
    if os.name == "nt":
        assert atomic.O_BINARY != 0
