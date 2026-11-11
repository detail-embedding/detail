"""Machine-tailored install sequences (§1): the per-OS scripts, real-path detection, the
terminal-launch argv, and that the invariant command parts stay in step with
docs/getting-started.md (the drift guard)."""

from pathlib import Path

from detail.checks.commands import (
    DETAIL_EXTRAS,
    SAM3_REPO_URL,
    current_script,
    detail_repo_root,
    setup_scripts,
    terminal_launch_argv,
    terminal_name,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GETTING_STARTED = _REPO_ROOT / "docs" / "getting-started.md"
_README = _REPO_ROOT / "README.md"


def _by_label(label):
    return next(s for s in setup_scripts() if s.os_label == label)


def test_setup_scripts_are_both_oses_current_first():
    scripts = setup_scripts()
    assert {s.os_label for s in scripts} == {"macOS / Linux", "Windows"}
    assert scripts[0].is_current  # the running OS is listed first
    assert sum(s.is_current for s in scripts) == 1


def test_current_script_is_a_full_sequence():
    commands = current_script().commands
    assert commands[0].startswith("cd ")
    assert "activate" in commands[1].lower()  # activate the environment
    assert f'uv pip install -e ".[{DETAIL_EXTRAS}]"' in commands
    assert any(c.startswith(f"git clone {SAM3_REPO_URL} ") for c in commands)
    assert any(
        c.startswith("uv pip install -e ") and "--no-deps" in c for c in commands
    )
    assert 'uv pip install "setuptools<81"' in commands
    assert commands[-1] == "detail model check sam3"  # ends by verifying


def test_windows_script_has_triton_and_shell_agnostic_activate():
    win = _by_label("Windows").commands
    mac = _by_label("macOS / Linux").commands
    # Extensionless, so the line works in PowerShell *and* cmd.exe. Activate.ps1 is
    # PowerShell-only and the default execution policy can block it.
    assert any(c.endswith("\\Scripts\\Activate") for c in win)
    assert not any("Activate.ps1" in c for c in win)
    assert "uv pip install triton-windows" in win
    assert "uv pip install triton-windows" not in mac
    assert any(c.startswith("source ") and "activate" in c for c in mac)


def test_current_os_uses_the_real_repo_path():
    # Running from a source checkout: the running OS's sequence carries absolute real paths.
    repo = detail_repo_root()
    assert repo is not None  # this test suite runs from an editable install
    commands = current_script().commands
    assert commands[0] == f"cd {repo}"
    clone = next(c for c in commands if c.startswith(f"git clone {SAM3_REPO_URL} "))
    assert str(repo.parent / "sam3") in clone  # sibling of the repo, absolute


def test_other_os_uses_relative_placeholder_forms():
    # The non-running OS can't know its absolute paths, so it shows the relative docs forms.
    other = next(s for s in setup_scripts() if not s.is_current)
    assert other.commands[0] == "cd <your detail folder>"
    assert any("sam3" in c and ".." in c for c in other.commands)  # ../sam3 or ..\sam3


def test_terminal_launch_argv_per_platform():
    assert terminal_launch_argv("/x", platform="darwin") == [
        "open",
        "-a",
        "Terminal",
        "/x",
    ]
    win = terminal_launch_argv("C:/x", platform="win32")
    assert win is not None and win[:3] == ["cmd", "/c", "start"]
    assert "C:/x" in " ".join(win)
    assert terminal_launch_argv("/x", platform="linux") is None  # no reliable command


def test_terminal_name_per_platform():
    assert terminal_name("darwin") == "Terminal"
    assert terminal_name("win32") == "PowerShell"
    assert terminal_name("linux") == "terminal"


def test_detail_repo_root_is_the_source_checkout():
    repo = detail_repo_root()
    assert repo is not None and (repo / "pyproject.toml").exists()


def test_invariant_command_parts_match_the_docs():
    """The parts of the commands that don't depend on machine paths (repo URL, extras, the
    setuptools cap, the Windows Triton line) must appear in getting-started.md, so the
    generated commands and the written install guide can't drift apart."""
    docs = _GETTING_STARTED.read_text(encoding="utf-8-sig")
    for fragment in (
        SAM3_REPO_URL,
        f".[{DETAIL_EXTRAS}]",
        'uv pip install "setuptools<81"',
        "uv pip install triton-windows",
        "\\Scripts\\Activate",
    ):
        assert fragment in docs, fragment


def test_windows_activate_is_extensionless_everywhere_it_is_written():
    """The Windows activation line is written in three places — the generated script, the
    install guide, and the README. All three shipped `Activate.ps1`, which only PowerShell
    runs and which the default execution policy can refuse."""
    written = [
        _GETTING_STARTED.read_text(encoding="utf-8-sig"),
        _README.read_text(encoding="utf-8-sig"),
        "\n".join(_by_label("Windows").commands),
    ]
    for text in written:
        assert "\\Scripts\\Activate" in text
        assert "Activate.ps1" not in text
