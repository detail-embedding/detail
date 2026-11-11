"""``scripts/run_python.sh`` finds a working Python on Linux, macOS and Windows, or stays quiet.

**The class: a command that names one interpreter.** The repo's ``statusLine`` and hook first
said ``python3``. That exists on Linux and macOS; on Windows the usual names are ``python`` and
``py``, and ``python3`` is typically a Microsoft Store stub that prints an error and exits
non-zero. Nothing fails on the machine it is written on. On Windows the status line is blank and
the hook errors on every pytest command.

These tests build a PATH that looks like each machine, since CI is Linux-only:
``test_a_store_stub_python3_is_skipped`` is the Windows case, ``test_py_launcher_is_the_last_resort``
the Windows machine with only the launcher, ``test_no_python_is_silent`` the machine with none
(silent, exit 0, so no hook error), and ``test_a_repo_path_with_spaces_works`` the common Windows
home directory. The settings file's commands going through this launcher is pinned in
``test_pytest_reminder_hook.py`` and here, ``test_the_settings_use_the_launcher``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "scripts" / "run_python.sh"

pytestmark = pytest.mark.skipif(os.name == "nt", reason="exercises sh; CI is Linux")

_WHICH = "import sys; print(sys.argv[1] if len(sys.argv) > 1 else '', sys.executable)"


def _bin(tmp_path: Path, **tools: str) -> Path:
    """A directory for PATH holding only the named tools, each a tiny sh script."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    for name, body in tools.items():
        tool = d / name
        tool.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        tool.chmod(0o755)
    return d


def _real(tag: str) -> str:
    """A tool that runs the real interpreter and says which name was used."""
    return f'if [ "$1" = -3 ]; then shift; fi; exec {sys.executable} "$@" {tag}'


def _run(
    tmp_path: Path, path: Path, root: Path | None = None
) -> subprocess.CompletedProcess:
    script = tmp_path / "which.py"
    script.write_text(_WHICH, encoding="utf-8")
    env = {
        "PATH": f"{path}:/usr/bin:/bin",
        "CLAUDE_PROJECT_DIR": str(root or tmp_path / "repo"),
    }
    return subprocess.run(
        [shutil.which("sh"), str(LAUNCHER), str(script)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )


def test_a_store_stub_python3_is_skipped(tmp_path):
    stub = "echo 'Python was not found; run without arguments to install from the Store' >&2; exit 9009"
    out = _run(tmp_path, _bin(tmp_path, python3=stub, python=_real("python")))
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("python "), out.stdout


def test_py_launcher_is_the_last_resort(tmp_path):
    stub = "exit 9009"
    out = _run(tmp_path, _bin(tmp_path, python3=stub, python=stub, py=_real("py")))
    assert out.stdout.startswith("py "), (out.stdout, out.stderr)


def test_the_repo_venv_comes_first(tmp_path):
    root = tmp_path / "repo"
    venv = root / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    out = _run(tmp_path, _bin(tmp_path, python3=_real("python3")), root=root)
    # Run directly, not through a PATH tool, so no name tag: just the venv's interpreter.
    assert out.stdout.strip() == str(venv / "python"), (out.stdout, out.stderr)


def test_a_repo_path_with_spaces_works(tmp_path):
    root = tmp_path / "First Last" / "repo"
    venv = root / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "python.exe").symlink_to(sys.executable)
    out = _run(tmp_path, _bin(tmp_path), root=root)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() != "", out.stderr


def test_no_python_is_silent(tmp_path):
    stub = "exit 9009"
    env = {
        "PATH": str(_bin(tmp_path, python3=stub, python=stub, py=stub)),
        "CLAUDE_PROJECT_DIR": str(tmp_path / "repo"),
    }
    out = subprocess.run(
        [shutil.which("sh"), str(LAUNCHER), "anything.py"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert (out.returncode, out.stdout, out.stderr) == (0, "", "")


def test_the_settings_use_the_launcher():
    settings = json.loads(
        (REPO / ".claude" / "settings.json").read_text(encoding="utf-8-sig")
    )
    commands = [settings["statusLine"]["command"]] + [
        h["command"] for e in settings["hooks"]["PreToolUse"] for h in e["hooks"]
    ]
    for command in commands:
        assert command.startswith(
            'sh "${CLAUDE_PROJECT_DIR:-.}/scripts/run_python.sh" '
        ), command
        assert "python3 " not in command


def test_the_launcher_is_checked_out_with_lf_endings():
    attrs = (REPO / ".gitattributes").read_text(encoding="utf-8-sig").splitlines()
    assert "*.sh text eol=lf" in attrs
    assert b"\r\n" not in LAUNCHER.read_bytes()
