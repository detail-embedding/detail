"""Machine-tailored install sequences (§1): the per-OS scripts, real-path detection, the
terminal-launch argv, and that the invariant command parts stay in step with
docs/getting-started.md (the drift guard)."""

from pathlib import Path

import pytest

from detail.checks import commands as commands_module
from detail.checks.commands import (
    DETAIL_EXTRAS,
    SAM3_REPO_URL,
    current_script,
    detail_repo_root,
    pip_command,
    setup_scripts,
    ssh_refresh_argv,
    ssh_terminal_supported,
    terminal_launch_argv,
    terminal_name,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GETTING_STARTED = _REPO_ROOT / "docs" / "getting-started.md"
_README = _REPO_ROOT / "README.md"


@pytest.fixture(autouse=True)
def _uv_present(monkeypatch):
    """docs/getting-started.md's own first step is "install uv" (§Install), so the
    generated commands this test file otherwise checks against it must reflect a
    machine that has one — never whatever the box actually running the suite
    happens to have installed. The "no uv" fallback gets its own tests below,
    which override this back to absent."""
    monkeypatch.setattr(commands_module.shutil, "which", lambda name: "/usr/bin/uv")


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


def test_pip_command_prefers_uv_when_present(monkeypatch):
    monkeypatch.setattr(commands_module.shutil, "which", lambda name: "/usr/bin/uv")
    assert pip_command() == "uv pip"


def test_pip_command_falls_back_to_plain_pip_when_uv_is_absent(monkeypatch):
    monkeypatch.setattr(commands_module.shutil, "which", lambda name: None)
    assert pip_command() == "pip"


def test_pip_command_never_guesses_uv_for_a_different_machine(monkeypatch):
    """is_current=False is the "docs form" block for the OS that isn't running (see
    setup_scripts) -- this process cannot know whether *that* machine has uv, and
    guessing it does is the wrong default. Even with uv on THIS machine's PATH, the
    non-current answer stays plain pip."""
    monkeypatch.setattr(commands_module.shutil, "which", lambda name: "/usr/bin/uv")
    assert pip_command(is_current=False) == "pip"


def test_setup_scripts_fall_back_to_plain_pip_without_uv(monkeypatch):
    """A working setup that never installed uv (a plain `python -m venv` + pip, conda,
    an HPC module environment) must get a copy-paste sequence that actually runs --
    found live: every command in the generated SAM 3 backend setup failed
    `uv: command not found` on a venv built without it."""
    monkeypatch.setattr(commands_module.shutil, "which", lambda name: None)
    commands = current_script().commands
    assert f'pip install -e ".[{DETAIL_EXTRAS}]"' in commands
    assert any(c.startswith("pip install -e ") and "--no-deps" in c for c in commands)
    assert 'pip install "setuptools<81"' in commands
    assert not any(c.startswith("uv ") for c in commands)
    win = _by_label("Windows").commands
    assert "pip install triton-windows" in win
    assert not any(c.startswith("uv ") for c in win)


def test_ssh_refresh_argv_darwin():
    argv = ssh_refresh_argv("tillicum", platform="darwin")
    assert argv[0] == "osascript"
    assert "ssh tillicum" in argv[-1]


def test_ssh_refresh_argv_windows():
    argv = ssh_refresh_argv("tillicum", platform="win32")
    assert argv is not None and argv[:3] == ["cmd", "/c", "start"]
    assert "ssh tillicum" in " ".join(argv)


def test_ssh_refresh_argv_plain_linux_is_unsupported(monkeypatch):
    monkeypatch.setattr(commands_module, "_is_wsl", lambda: False)
    assert ssh_refresh_argv("tillicum", platform="linux") is None


def test_ssh_refresh_argv_wsl_reenters_the_same_distro(monkeypatch):
    """WSL interop puts Windows executables on PATH, but the SSH config/agent/
    known_hosts this needs live inside the WSL distro, not on Windows -- so the
    opened terminal must come back into WSL (`wsl.exe -e ...`), not just run a
    bare `ssh` from PowerShell (a different, unconfigured client)."""
    monkeypatch.setattr(commands_module, "_is_wsl", lambda: True)
    monkeypatch.setattr(commands_module.shutil, "which", lambda name: None)
    argv = ssh_refresh_argv("tillicum", platform="linux")
    assert argv[0] == "powershell.exe"
    joined = " ".join(argv)
    assert "wsl.exe" in joined
    assert "ssh tillicum" in joined


def _wsl_script(
    monkeypatch, *, distro="Ubuntu", which="/mnt/c/Windows/system32/wsl.exe"
):
    """The PowerShell the WSL strategy opens, with the three things it reads from the machine
    (the distro, `wsl.exe` on PATH, and `wslpath`) pinned."""
    monkeypatch.setattr(commands_module, "_is_wsl", lambda: True)
    if distro is None:
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    else:
        monkeypatch.setenv("WSL_DISTRO_NAME", distro)
    monkeypatch.setattr(
        commands_module.shutil,
        "which",
        lambda name: which if name == "wsl.exe" else None,
    )
    monkeypatch.setattr(
        commands_module,
        "_windows_path",
        lambda p: (
            r"C:\Windows\system32\wsl.exe" if str(p).endswith("wsl.exe") else None
        ),
    )
    return ssh_refresh_argv("tillicum", platform="linux")[-1]


def test_the_wsl_window_enters_this_distro_not_the_default_one(monkeypatch):
    """`wsl.exe -e` alone enters the *default* distro, which need not be the one holding the SSH
    config, agent and known_hosts this exists to reach."""
    assert "'-d' 'Ubuntu-24.04'" in _wsl_script(monkeypatch, distro="Ubuntu-24.04")
    assert "'-d'" not in _wsl_script(monkeypatch, distro=None)


def test_the_wsl_window_names_wsl_exe_by_its_windows_path(monkeypatch):
    """PowerShell searching its own PATH found no usable `wsl.exe` on one machine (two copies
    are installed side by side there); the one on this process's PATH is passed explicitly.
    """
    script = _wsl_script(monkeypatch)
    assert script.startswith("try { & 'C:\\Windows\\system32\\wsl.exe' ")
    # no `wsl.exe` on this process's PATH, or no translation for it: the bare name, as before
    assert "& 'wsl.exe' " in _wsl_script(monkeypatch, which=None)
    monkeypatch.setattr(commands_module, "_windows_path", lambda p: None)
    assert "& 'wsl.exe' " in ssh_refresh_argv("tillicum", platform="linux")[-1]


def test_the_wsl_window_says_what_to_do_when_wsl_cannot_start(monkeypatch):
    """Without this the window ends on `Program 'wsl.exe' failed to run: Access is denied` and
    nothing else, which says nothing about what still works."""
    script = _wsl_script(monkeypatch)
    assert "catch {" in script
    assert "Could not start WSL from this window: " in script
    assert "Open your WSL terminal and run:  ssh tillicum" in script


def test_a_host_with_a_space_or_a_quote_survives_both_layers_of_quoting(monkeypatch):
    """The host is quoted for bash, and that whole command is quoted again for PowerShell."""
    monkeypatch.setattr(commands_module, "_is_wsl", lambda: True)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(commands_module.shutil, "which", lambda name: None)
    script = ssh_refresh_argv("my host", platform="linux")[-1]
    assert "'ssh ''my host'''" in script
    quoted = ssh_refresh_argv("a'b", platform="linux")[-1]
    assert "'ssh ''a''\"''\"''b'''" in quoted


def test_ssh_terminal_supported_matches_ssh_refresh_argv(monkeypatch):
    for platform, wsl in (("darwin", False), ("win32", False), ("linux", False)):
        monkeypatch.setattr(commands_module, "_is_wsl", lambda: wsl)
        assert ssh_terminal_supported(platform=platform) == (
            ssh_refresh_argv("x", platform=platform) is not None
        )
    monkeypatch.setattr(commands_module, "_is_wsl", lambda: True)
    assert ssh_terminal_supported(platform="linux") is True
    assert ssh_refresh_argv("tillicum", platform="linux") is not None


def test_wsl_explorer_argv_is_none_off_wsl(monkeypatch):
    monkeypatch.setattr(commands_module, "_is_wsl", lambda: False)
    assert commands_module.wsl_explorer_argv("/home/u/exp") is None


def test_wsl_explorer_argv_translates_the_path_for_explorer(monkeypatch):
    """Explorer is a Windows program: it needs the Windows spelling of the folder."""
    import subprocess

    monkeypatch.setattr(commands_module, "_is_wsl", lambda: True)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="\\\\wsl$\\Ubuntu\\x\n")

    monkeypatch.setattr(commands_module.subprocess, "run", fake_run)
    assert commands_module.wsl_explorer_argv("/home/u/x") == [
        "explorer.exe",
        "\\\\wsl$\\Ubuntu\\x",
    ]
    assert calls == [["wslpath", "-w", "/home/u/x"]]


def test_wsl_explorer_argv_is_none_when_wslpath_fails(monkeypatch):
    """No translation means no fallback, so the caller keeps its own failure message."""
    monkeypatch.setattr(commands_module, "_is_wsl", lambda: True)

    def missing(argv, **kwargs):
        raise FileNotFoundError("wslpath")

    monkeypatch.setattr(commands_module.subprocess, "run", missing)
    assert commands_module.wsl_explorer_argv("/home/u/x") is None


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
