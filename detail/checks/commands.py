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
            f'uv pip install -e ".[{DETAIL_EXTRAS}]"',
            "Install detail and the models it needs",
        ),
        SetupStep(
            f"git clone {SAM3_REPO_URL} {sam3}",
            "Download the SAM 3 source (Meta ships it off PyPI)",
        ),
        SetupStep(f"uv pip install -e {sam3} --no-deps", "Install the SAM 3 package"),
        SetupStep(
            'uv pip install "setuptools<81"', "Pin a setuptools the SAM 3 build needs"
        ),
    ]
    if windows:
        steps.append(
            SetupStep("uv pip install triton-windows", "Install Triton (Windows only)")
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
