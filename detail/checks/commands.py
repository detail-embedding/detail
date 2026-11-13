"""Machine-tailored, copy-paste install sequences — the actionable side of a failing check.

A failing check (``api/checks``) says *what* is missing; this module says *how to fix it*
with the complete command sequence to run, from opening a terminal to verifying. It uses the
paths the running tool actually knows — the detail source tree and the active environment — so
the commands are ready to paste, not templates to fill in. Two sequences are produced, one per
OS (macOS/Linux and Windows PowerShell): the sequence for the OS the tool is running on carries
real absolute paths; the other OS's carries the relative forms (it can't know that machine's
folders), clearly the docs form.

Pure and Qt-free (``os`` / ``sys`` / ``pathlib``, no backend imports), so both the CLI
(``detail doctor``) and the GUI (``Sam3BackendPage``) reach it through ``api/checks``. The
invariant parts of the commands (repo URL, extras, the setuptools cap, the Windows Triton line)
mirror ``docs/getting-started.md`` — a drift test keeps them in step.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SAM3_REPO_URL = "https://github.com/facebookresearch/sam3"
# Kept in sync with docs/getting-started.md § Install. The full set the tool needs to run.
DETAIL_EXTRAS = "yolo,torch,gui,analysis,sam3"

_REPO_PLACEHOLDER = "<your detail folder>"


@dataclass(frozen=True)
class SetupStep:
    """One command plus a concise description of what it does."""

    command: str
    note: str


@dataclass(frozen=True)
class SetupScript:
    """A full, ordered install sequence for one OS — from opening a terminal to verifying."""

    os_label: str
    is_current: bool  # the OS the tool is running on (real paths + the terminal button)
    steps: tuple[SetupStep, ...]

    @property
    def commands(self) -> tuple[str, ...]:
        return tuple(step.command for step in self.steps)

    def as_script(self) -> str:
        """The sequence as a copy-paste block, each command under a ``#`` description.

        ``#`` is a comment in both bash and PowerShell, so the block pastes and runs as-is.
        """
        lines: list[str] = []
        for step in self.steps:
            lines.append(f"# {step.note}")
            lines.append(step.command)
        return "\n".join(lines)


def detail_repo_root() -> Path | None:
    """The detail source tree (an editable install), or None if installed non-editably.

    ``detail/checks/commands.py`` sits two levels under the repo root; the ``pyproject.toml``
    check rejects a site-packages location (a non-editable install has no repo to point at).
    """
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").exists() else None


def active_env_dir() -> Path | None:
    """The active environment's directory when it is a virtualenv / uv venv, else None."""
    prefix = Path(sys.prefix)
    return prefix if (prefix / "pyvenv.cfg").exists() else None


def pip_command(is_current: bool = True) -> str:
    """``uv pip`` if this machine has ``uv`` on ``PATH``, else plain ``pip``.

    A generated command has to run on the machine it is generated for — a copy-paste
    sequence that hardcodes ``uv pip`` fails ``uv: command not found`` at its very first
    step on any of the plenty of working setups (a plain ``python -m venv`` + ``pip``,
    conda, an HPC module environment) that never installed ``uv`` at all. Checked once
    here rather than at each call site, so the CUDA doctor's reinstall command
    (``gui/cuda_doctor.py``) and this module's install sequences can't drift into two
    different answers for the same machine.

    ``is_current=False`` (the "docs form" block for the OS that isn't running, see
    :func:`setup_scripts`) always answers ``pip``: this process cannot know whether a
    *different* machine has ``uv``, and guessing it does is the wrong default — ``pip``
    is the one thing every Python environment actually has.
    """
    if is_current and shutil.which("uv") is not None:
        return "uv pip"
    return "pip"


def _activate_command(windows: bool, env: str) -> str:
    # Extensionless: PowerShell infers .ps1 and cmd.exe finds Activate.bat via PATHEXT, so
    # one line works in either shell. Naming Activate.ps1 outright is PowerShell-only, and
    # the default execution policy can refuse to run it.
    if windows:
        return f"{env}\\Scripts\\Activate"
    return f"source {env}/bin/activate"


def _script(
    *, windows: bool, is_current: bool, repo: Path | None, env: Path | None
) -> SetupScript:
    real = is_current and repo is not None
    sep = "\\" if windows else "/"
    pip = pip_command(is_current)
    if real:
        cd_target = str(repo)
        sam3 = str(repo.parent / "sam3")
        env_dir = str(env) if env is not None else f"{cd_target}{sep}.venv"
    else:
        # The non-running OS (or a non-editable install): its absolute paths are unknowable,
        # so use the relative docs forms and a placeholder folder to cd into.
        cd_target = _REPO_PLACEHOLDER
        sam3 = f"..{sep}sam3"
        env_dir = ".venv"
    steps = [
        SetupStep(f"cd {cd_target}", "Go to your detail folder"),
        SetupStep(
            _activate_command(windows, env_dir), "Activate the detail environment"
        ),
        SetupStep(
            f'{pip} install -e ".[{DETAIL_EXTRAS}]"',
            "Install detail and the models it needs",
        ),
        SetupStep(
            f"git clone {SAM3_REPO_URL} {sam3}",
            "Download the SAM 3 source (Meta ships it off PyPI)",
        ),
        SetupStep(f"{pip} install -e {sam3} --no-deps", "Install the SAM 3 package"),
        SetupStep(
            f'{pip} install "setuptools<81"', "Pin a setuptools the SAM 3 build needs"
        ),
    ]
    if windows:
        steps.append(
            SetupStep(f"{pip} install triton-windows", "Install Triton (Windows only)")
        )
    steps.append(SetupStep("detail model check sam3", "Check that SAM 3 is installed"))
    os_label = "Windows" if windows else "macOS / Linux"
    return SetupScript(os_label, is_current, tuple(steps))


def setup_scripts() -> list[SetupScript]:
    """The full install sequence for each OS, the running OS first (with real paths)."""
    repo = detail_repo_root()
    env = active_env_dir()
    current_windows = os.name == "nt"
    scripts = [
        _script(windows=False, is_current=not current_windows, repo=repo, env=env),
        _script(windows=True, is_current=current_windows, repo=repo, env=env),
    ]
    scripts.sort(key=lambda s: not s.is_current)
    return scripts


def current_script() -> SetupScript:
    """The install sequence for the OS the tool is running on."""
    return next(s for s in setup_scripts() if s.is_current)


def terminal_name(platform: str | None = None) -> str:
    """The terminal app's name on this platform, for the Open-terminal button label."""
    name = platform if platform is not None else sys.platform
    if name == "darwin":
        return "Terminal"
    if name.startswith("win"):
        return "PowerShell"
    return "terminal"


def terminal_launch_argv(
    repo_dir: str, *, platform: str | None = None
) -> list[str] | None:
    """The argv to open a native terminal already in ``repo_dir``, or None if unsupported.

    macOS opens Terminal.app there; Windows opens a PowerShell window there. Linux has no
    single reliable command across desktops, so the caller should offer a note instead.
    """
    name = platform if platform is not None else sys.platform
    if name == "darwin":
        return ["open", "-a", "Terminal", repo_dir]
    if name.startswith("win"):
        return [
            "cmd",
            "/c",
            "start",
            "",
            "powershell",
            "-NoExit",
            "-Command",
            f"Set-Location -LiteralPath '{repo_dir}'",
        ]
    return None


def _is_wsl() -> bool:
    """Whether this process is Python running *inside* WSL — not native Linux, not
    native Windows. ``/proc/version`` naming ``microsoft`` is the standard signal
    both WSL1 and WSL2 report (there is no ``sys.platform`` value for it; both
    report plain ``"linux"``)."""
    try:
        return (
            "microsoft" in Path("/proc/version").read_text(encoding="utf-8-sig").lower()
        )
    except OSError:
        return False


def wsl_explorer_argv(path: Path | str) -> list[str] | None:
    """The argv that shows ``path`` in Windows Explorer from inside WSL, or ``None``.

    A WSL distro usually has no ``xdg-open`` handler for ``file://``, so Qt's
    ``QDesktopServices.openUrl`` finds no launcher and does nothing. ``None`` off WSL, and when
    ``wslpath`` cannot translate the path, so the caller keeps its own failure message.
    """
    if not _is_wsl():
        return None
    windows = _windows_path(path)
    return ["explorer.exe", windows] if windows else None


def wsl_browser_argv(url: str) -> list[str] | None:
    """The argv that opens web ``url`` in the Windows default browser from inside WSL, or
    ``None`` off WSL.

    The web twin of :func:`wsl_explorer_argv`: a WSL distro usually has no browser for
    ``xdg-open`` either, so ``QDesktopServices.openUrl`` returns ``False`` and Qt only logs
    *Unable to detect a web browser*. ``explorer.exe`` hands a URL to the Windows shell, which
    opens the default browser; unlike ``cmd.exe /c start`` it parses no ``&`` and needs no
    Windows working directory. A ``http://127.0.0.1`` URL (the offline docs) still reaches
    this distro, because WSL2 forwards localhost to Windows.
    """
    return ["explorer.exe", url] if _is_wsl() else None


def _windows_path(path: Path | str) -> str | None:
    """``path`` spelled the way a Windows program reads it (``wslpath -w``), or ``None``."""
    try:
        windows = subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return windows or None


def _powershell_quoted(text: str) -> str:
    """``text`` as a PowerShell single-quoted literal (a quote inside is doubled)."""
    return "'" + text.replace("'", "''") + "'"


def _wsl_session_script(host: str) -> str:
    """The PowerShell the new window runs to come back into this distro and ``ssh`` to ``host``.

    Three things it does not leave to chance, each for a reason seen on a real machine:

    * **This distro** (``-d $WSL_DISTRO_NAME``). A bare ``wsl.exe -e`` enters the *default*
      distro, which is not necessarily the one holding the SSH config, agent and known_hosts this
      exists to reach.
    * **A named ``wsl.exe``.** PowerShell searches its own PATH, which its profile may change and
      which here holds two copies (the inbox ``System32`` forwarder and the installed WSL);
      interop already put one on *this* process's PATH, so that one is passed by its Windows path.
    * **A message when it cannot start.** PowerShell's own error for a refused launch
      (``Program 'wsl.exe' failed to run: Access is denied``) says nothing about what to do next,
      and the window stays open on it; this prints the one thing that still works.
    """
    inner = f"ssh {shlex.quote(host)}"
    found = shutil.which("wsl.exe")
    wsl = (_windows_path(found) if found else None) or "wsl.exe"
    distro = os.environ.get("WSL_DISTRO_NAME")
    args = [*(["-d", distro] if distro else []), "-e", "bash", "-lic", inner]
    call = "& " + " ".join(_powershell_quoted(a) for a in [wsl, *args])
    advice = _powershell_quoted(f"Open your WSL terminal and run:  ssh {host}")
    return (
        f"try {{ {call} }} "
        "catch { Write-Host ''; "
        "Write-Host ('Could not start WSL from this window: ' + $_.Exception.Message); "
        f"Write-Host {advice} }}"
    )


def _terminal_kind(*, platform: str | None = None) -> str | None:
    """Which terminal-launch strategy this machine supports — ``"darwin"``,
    ``"windows"``, ``"wsl"``, or ``None`` where no reliable one exists (plain
    Linux, same gap as :func:`terminal_launch_argv`). The one place
    :func:`ssh_refresh_argv` and :func:`ssh_terminal_supported` both ask "can
    this even work here", so a caller can grey a button out up front (per
    CLAUDE.md's "grey out, don't hide") instead of only failing after the click.
    """
    name = platform if platform is not None else sys.platform
    if name == "darwin":
        return "darwin"
    if name.startswith("win"):
        return "windows"
    if name == "linux" and _is_wsl():
        return "wsl"
    return None


def ssh_terminal_supported(*, platform: str | None = None) -> bool:
    """Whether :func:`ssh_refresh_argv` can open a terminal on this machine at
    all — check once to grey a button out up front rather than only on click."""
    return _terminal_kind(platform=platform) is not None


def ssh_refresh_argv(host: str, *, platform: str | None = None) -> list[str] | None:
    """The argv to open a terminal running an interactive ``ssh <host>`` login — the
    one-click form of "run ``ssh <host>`` once to refresh your cached session"
    (§6): a Slurm cluster's SSH session (Kerberos ticket, a cached Duo push) times
    out independently of anything DETAIL does, and every remote command shells out
    with ``BatchMode=yes`` — it fails fast with "Permission denied" rather than
    prompting, so refreshing it needs one real, interactive ``ssh`` login somewhere.

    DETAIL never sees the password or the Duo approval — this only *opens* the
    terminal ``ssh`` itself already runs in; the login happens exactly as it would
    if the user had typed the command themselves (§6's "nothing here ever sees,
    stores, or prompts for a credential", one level up: at the GUI, not just the
    transport).

    macOS and Windows launch a native terminal directly. Plain Linux has no single
    reliable terminal command (same gap as :func:`terminal_launch_argv`) and
    returns ``None`` — except **WSL**, which this feature's entire audience runs
    under (the whole Tillicum workflow is a WSL guide): WSL interop puts Windows
    executables on this process's own ``PATH`` by default (``wsl.conf``'s
    ``[interop] appendWindowsPath``), so a real Windows terminal can be opened that
    re-enters *this* WSL distro — the SSH config, agent, and known_hosts this needs
    live inside WSL, not on Windows, so running ``ssh`` directly from PowerShell
    would be a different (and unconfigured) SSH client entirely.
    """
    kind = _terminal_kind(platform=platform)
    if kind == "darwin":
        script = f"ssh {host}".replace("\\", "\\\\").replace('"', '\\"')
        return [
            "osascript",
            "-e",
            f'tell application "Terminal" to do script "{script}"',
        ]
    if kind == "windows":
        return [
            "cmd",
            "/c",
            "start",
            "",
            "powershell",
            "-NoExit",
            "-Command",
            f"ssh {host}",
        ]
    if kind == "wsl":
        return ["powershell.exe", "-NoExit", "-Command", _wsl_session_script(host)]
    return None
