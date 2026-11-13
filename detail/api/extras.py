"""Install an optional extra's missing packages into the running environment (§1).

One of DETAIL's extras is plain PyPI packages with no build step and no GPU choice:
``analysis`` (SHAP, which Cluster profile's default explainer needs, beside scikit-learn and
the clustering libraries). That one this module installs for you — into **the interpreter that
is running DETAIL** (``sys.executable``), never whichever ``pip`` happens to be first on
``PATH``, which is how a package gets installed and still cannot be imported. Everything else
(PyTorch's CUDA wheel, the SAM 3 source build) stays a copy-paste command, because it needs a
choice this tool should not make for you.

- **What an extra holds** is read from ``pyproject.toml`` when DETAIL runs from its source tree
  (an editable install), else from the installed package's metadata. The editable install's
  metadata is written once, at install time, so it goes stale when an extra gains a package —
  ``shap`` joined ``analysis`` after most machines had installed it.
- **The installer** is ``uv pip install --python <this interpreter>`` where ``uv`` is on
  ``PATH`` (the test ``checks.commands.pip_command`` uses for every generated command, so the
  two cannot disagree), else ``<this interpreter> -m pip install``.
- **Progress** is the installer's own output, streamed: a phase per line it prints
  (resolving → downloading → installing), and a report at least every
  :data:`_HEARTBEAT_S` while it is quiet, so a Cancel lands promptly. Cancel kills the
  installer's process tree.
- **The running app sees the result without a restart**: after a successful install the
  site-packages ``.pth`` files are re-read and the import caches invalidated
  (``core.installed.refresh_site_packages``), so ``explainers.resolve`` picks SHAP on its next
  call. A compiled package the process already imported is the exception, and none of these is
  imported before it is needed.

Qt-free: the GUI (Configuration ▸ *Cluster profile*) and the CLI
(``detail install-extra``) both call :func:`install_extra`.
"""

from __future__ import annotations

import importlib.metadata
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass

from detail.checks import commands
from detail.core import installed
from detail.core import interpreter

#: The extras this tool installs for you: PyPI-only, no build step, no hardware choice.
INSTALLABLE: tuple[str, ...] = ("analysis",)
#: What each installable extra is for, said where the install is offered.
PURPOSE = {
    "analysis": "SHAP for Cluster profile, and the clustering libraries",
}
#: Longest quiet stretch, seconds, before the progress callback is called anyway.
_HEARTBEAT_S = 0.25
#: Lines of installer output kept for an error message.
_TAIL_LINES = 40

ProgressFn = Callable[[str, float, str], None]


class InstallError(RuntimeError):
    """The installer failed, or finished without installing what was missing. The message
    carries the tail of its output."""


@dataclass(frozen=True)
class ExtraStatus:
    """An extra's requirements, and which of them this environment lacks."""

    extra: str
    requirements: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def installed(self) -> bool:
        return not self.missing


def _name(requirement: str) -> str:
    """The distribution name of a requirement string (``numba>=0.60`` → ``numba``)."""
    return re.split(r"[\s<>=!~;\[(]", requirement.strip(), maxsplit=1)[0]


def _check(extra: str) -> None:
    if extra not in INSTALLABLE:
        raise ValueError(
            f"{extra!r} is not an extra this tool installs; it installs "
            f"{', '.join(INSTALLABLE)}"
        )


def requirements(extra: str) -> list[str]:
    """The requirement strings of ``extra``, as the source tree declares them.

    Raises:
        ValueError: If ``extra`` is not one this tool installs, or DETAIL declares no such
            extra.
    """
    _check(extra)
    repo = commands.detail_repo_root()
    if repo is not None:
        import tomllib

        with open(repo / "pyproject.toml", "rb") as f:
            extras = tomllib.load(f)["project"].get("optional-dependencies", {})
        if extra not in extras:
            raise ValueError(f"pyproject.toml declares no extra {extra!r}")
        return [str(r) for r in extras[extra]]
    marker = re.compile(rf"extra\s*==\s*['\"]{re.escape(extra)}['\"]")
    found = [
        r.split(";", 1)[0].strip()
        for r in importlib.metadata.requires("detail") or []
        if marker.search(r)
    ]
    if not found:
        raise ValueError(f"the installed detail declares no extra {extra!r}")
    return found


def _applies(requirement: str) -> bool:
    """Whether ``requirement``'s environment marker holds here (no marker: always)."""
    if ";" not in requirement:
        return True
    try:
        from packaging.markers import Marker
    except ImportError:  # no `packaging`: let the installer decide
        return True
    return Marker(requirement.split(";", 1)[1]).evaluate()


def _satisfied(requirement: str) -> bool:
    """Whether the installed distribution meets ``requirement`` (name and version)."""
    try:
        version = importlib.metadata.version(_name(requirement))
    except importlib.metadata.PackageNotFoundError:
        return False
    spec = requirement.split(";", 1)[0][len(_name(requirement)) :].strip()
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:  # present is enough without `packaging`
        return True
    return SpecifierSet(spec).contains(version, prereleases=True)


def status(extra: str) -> ExtraStatus:
    """Which of ``extra``'s packages this environment lacks. A disk check, no network."""
    reqs = tuple(r for r in requirements(extra) if _applies(r))
    return ExtraStatus(extra, reqs, tuple(r for r in reqs if not _satisfied(r)))


def package_version(name: str) -> str | None:
    """The installed version of distribution ``name``, or ``None`` when it is absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def install_argv(packages: list[str] | tuple[str, ...]) -> list[str]:
    """The installer command for ``packages`` into the running interpreter's environment."""
    uv = shutil.which("uv") if commands.pip_command() == "uv pip" else None
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable, *packages]
    return interpreter.module_argv("pip", "install", *packages)


def _quote(arg: str, *, windows: bool) -> str:
    if arg and not re.search(r"""[\s<>|&^()'";$`*?!]""", arg):
        return arg
    if windows:
        return '"' + arg.replace('"', '`"') + '"'
    return "'" + arg.replace("'", "'\"'\"'") + "'"


def command_text(extra: str) -> str:
    """The install as one line to paste into this machine's terminal (PowerShell on
    Windows), for when the button cannot be used. Names the running interpreter, so it
    installs where the app will find it whichever environment the terminal has active.
    """
    current = status(extra)
    packages = list(current.missing or current.requirements)
    argv = install_argv(packages)
    windows = os.name == "nt"
    words = [_quote(a, windows=windows) for a in argv]
    if windows and words[0].startswith('"'):
        words[0] = "& " + words[0]  # PowerShell runs a quoted path only after `&`
    if argv[0] != sys.executable:  # uv: show it by name, as the docs do
        words[0] = "uv"
    return " ".join(words)


#: The installer's lines that mark a phase, in order, with the fraction each starts at.
_PHASES = (
    (
        re.compile(r"^(Collecting|Resolved|Using cached|Requirement already)"),
        "resolving",
        0.1,
    ),
    (re.compile(r"^(Downloading|Downloaded|Prepared|Building)"), "downloading", 0.4),
    (
        re.compile(r"^(Installing collected|Uninstall|Installed|Successfully)"),
        "installing",
        0.8,
    ),
)


def _phase(line: str, current: tuple[str, float]) -> tuple[str, float]:
    """The phase ``line`` moves to — never backwards, and creeping forward within one, so a
    long resolve still shows the bar move."""
    stage, fraction = current
    text = line.strip().lstrip("+ ").strip()
    for pattern, name, start in _PHASES:
        if pattern.match(text) and start >= fraction:
            return name, start
    nxt = next((s for _p, _n, s in _PHASES if s > fraction), 0.95)
    return stage, min(fraction + 0.01, nxt - 0.01)


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True
        )
    else:
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    proc.wait()


def run_installer(argv: list[str], progress: ProgressFn | None = None) -> list[str]:
    """Run ``argv``, streaming its output into ``progress``; return every line it printed.

    An exception from ``progress`` (the GUI's Cancel raises one) kills the installer's
    process tree before it propagates.

    Raises:
        InstallError: The installer could not start, or exited non-zero.
    """
    from detail.executor.local import new_process_group_kwargs

    def emit(stage: str, fraction: float, detail: str) -> None:
        if progress is not None:
            progress(stage, fraction, detail)

    emit("starting", 0.0, " ".join(argv[:3]) + " …")
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            # uv writes UTF-8; pip writes what PYTHONIOENCODING says, so say UTF-8 to both
            # ends. A stray byte becomes U+FFFD rather than ending the read.
            encoding="utf-8",
            errors="replace",
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "PIP_NO_INPUT": "1",
            },
            **new_process_group_kwargs(),
        )
    except OSError as exc:
        raise InstallError(f"could not start the installer ({argv[0]}): {exc}") from exc
    lines: list[str] = []
    feed: queue.Queue = queue.Queue()

    def read() -> None:
        for raw in proc.stdout:
            feed.put(raw.rstrip())
        feed.put(None)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    phase = ("starting", 0.0)
    last = ""
    try:
        while True:
            try:
                line = feed.get(timeout=_HEARTBEAT_S)
            except queue.Empty:
                emit(phase[0], phase[1], last)  # quiet: still lets a Cancel land
                continue
            if line is None:
                break
            if not line.strip():
                continue
            lines.append(line)
            last = line.strip()
            phase = _phase(line, phase)
            emit(phase[0], phase[1], last)
        code = proc.wait()
    except BaseException:
        _kill(proc)
        raise
    finally:
        # The pipe is the reader's until it sees EOF, which a finished or killed installer
        # sends; close it after, or it is left for the garbage collector to complain about.
        reader.join(timeout=5)
        proc.stdout.close()
    if code != 0:
        tail = "\n".join(lines[-_TAIL_LINES:])
        raise InstallError(
            f"the installer exited with code {code}:\n{' '.join(argv)}\n\n{tail}"
        )
    return lines


def install_extra(extra: str, progress: ProgressFn | None = None) -> ExtraStatus:
    """Install ``extra``'s missing packages into the running environment; return its status
    after, re-read so this process sees the new packages without a restart.

    Raises:
        ValueError: ``extra`` is not one this tool installs.
        InstallError: The installer failed (its output tail is in the message), or finished
            and the packages are still missing.
    """
    before = status(extra)
    if before.installed:
        if progress is not None:
            progress("installed", 1.0, f"{extra}: everything is already installed")
        return before
    lines = run_installer(install_argv(before.missing), progress)
    installed.refresh_site_packages()
    after = status(extra)
    if after.missing:
        tail = "\n".join(lines[-_TAIL_LINES:])
        raise InstallError(
            f"the installer finished, but {', '.join(after.missing)} is still missing from "
            f"{sys.executable}'s environment:\n\n{tail}"
        )
    if progress is not None:
        progress("installed", 1.0, f"installed {', '.join(before.missing)}")
    return after
