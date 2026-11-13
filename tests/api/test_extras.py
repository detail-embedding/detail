"""Installing an optional extra into the running environment (``api/extras.py``, §1).

The guards:

- **It installs into the interpreter running DETAIL**, by ``uv pip --python <it>`` or
  ``<it> -m pip`` — never whichever ``pip`` is first on ``PATH``.
- **An extra's packages come from the source tree's ``pyproject.toml``**, whose ``analysis``
  holds ``shap``; an editable install's metadata predates it.
- **The installer's output is the progress, a failure carries its tail, and Cancel kills it.**
  Run against a stand-in installer — a real subprocess printing what pip prints and writing a
  package into a directory on ``sys.path`` — never a real install.
- **The running process sees the new package without a restart**, so Cluster profile's
  ``explainers.resolve`` picks SHAP on its next call instead of falling back.
"""

from __future__ import annotations

import dataclasses
import sys
import time
import tomllib
import uuid
from pathlib import Path

import pytest

from detail.analysis import explainers
from detail.api import extras
from detail.checks import commands
from detail.core import installed
from detail.core.progress import OperationCancelled


def _fake_installer(
    target: Path,
    module: str,
    *,
    fail: bool = False,
    hang: bool = False,
    editable: bool = False,
):
    """``argv`` for a stand-in installer: prints pip's phases, then writes ``module`` (a
    package and its dist-info) into ``target``, or fails, or hangs silently. ``editable``
    puts the package beside ``target`` and a ``.pth`` naming it inside, the way an editable
    install does — visible to a running process only once it re-reads its ``.pth`` files.
    """
    dist = module.replace("_", "-")
    script = f"""
import os, sys, time
from pathlib import Path
print("Collecting {dist}", flush=True)
print("  Downloading {dist}-1.0-py3-none-any.whl (12 kB)", flush=True)
if {hang!r}:
    Path(r"{target}", "pid").write_text(str(os.getpid()))
    time.sleep(60)
if {fail!r}:
    print("ERROR: No matching distribution found for {dist}", flush=True)
    sys.exit(1)
print("Installing collected packages: {dist}", flush=True)
home = Path(r"{target}")
if {editable!r}:
    home = Path(r"{target}").parent / "src"
    home.mkdir()
    Path(r"{target}", "__editable__.{module}.pth").write_text(str(home) + "\\n")
pkg = home / "{module}"
pkg.mkdir()
(pkg / "__init__.py").write_text("")
meta = Path(r"{target}", "{module}-1.0.dist-info")
meta.mkdir()
(meta / "METADATA").write_text("Metadata-Version: 2.1\\nName: {dist}\\nVersion: 1.0\\n")
print("Successfully installed {dist}-1.0", flush=True)
"""
    return [sys.executable, "-c", script]


@pytest.fixture
def fake_extra(tmp_path, monkeypatch):
    """An extra holding one package nothing has installed, and a directory on ``sys.path``
    — already scanned, so its new contents are invisible to a cached finder — to install
    it into."""
    module = f"detail_fake_{uuid.uuid4().hex[:8]}"
    target = tmp_path / "site"
    target.mkdir()
    monkeypatch.syspath_prepend(str(target))
    assert (
        installed.importlib.util.find_spec(module) is None
    )  # the finder has looked here
    monkeypatch.setattr(
        extras, "requirements", lambda extra: [module.replace("_", "-")]
    )
    return module, target


def test_the_analysis_extra_is_read_from_the_source_tree():
    repo = commands.detail_repo_root()
    with open(repo / "pyproject.toml", "rb") as f:
        declared = tomllib.load(f)["project"]["optional-dependencies"]["analysis"]
    assert extras.requirements("analysis") == declared
    assert "shap" in [extras._name(r) for r in declared]


def test_only_the_plain_pypi_extras_are_installed_for_you():
    with pytest.raises(ValueError, match="not an extra this tool installs"):
        extras.requirements("torch")


def test_it_installs_into_the_running_interpreter(monkeypatch):
    monkeypatch.setattr(commands, "pip_command", lambda is_current=True: "pip")
    assert extras.install_argv(["shap"])[:3] == [sys.executable, "-m", "pip"]
    monkeypatch.setattr(commands, "pip_command", lambda is_current=True: "uv pip")
    monkeypatch.setattr(extras.shutil, "which", lambda name: "/bin/uv")
    argv = extras.install_argv(["shap"])
    assert argv[:3] == ["/bin/uv", "pip", "install"]
    assert argv[argv.index("--python") + 1] == sys.executable


def test_the_command_to_paste_quotes_what_a_shell_would_split(monkeypatch):
    monkeypatch.setattr(commands, "pip_command", lambda is_current=True: "pip")
    monkeypatch.setattr(
        extras,
        "status",
        lambda extra: extras.ExtraStatus(extra, ("numba>=0.60",), ("numba>=0.60",)),
    )
    text = extras.command_text("analysis")
    # Unquoted, `>=0.60` is a redirect in bash and PowerShell alike: it would install numba
    # unpinned and write a file called `=0.60`.
    assert ("'numba>=0.60'" in text) or ('"numba>=0.60"' in text)


def test_the_install_streams_its_phases_and_the_app_sees_the_package(
    fake_extra, monkeypatch
):
    module, target = fake_extra
    monkeypatch.setattr(
        extras, "install_argv", lambda pkgs: _fake_installer(target, module)
    )
    # Cluster profile's default explainer, needing the package being installed — the same
    # registry entry `resolve` reads, with only its import name swapped for the fake one.
    shap_like = dataclasses.replace(
        explainers.EXPLAINERS[explainers.DEFAULT_EXPLAINER], needs=(module,)
    )
    fallback = dataclasses.replace(
        explainers.EXPLAINERS[explainers.FALLBACK_EXPLAINER], needs=()
    )
    monkeypatch.setitem(explainers.EXPLAINERS, explainers.DEFAULT_EXPLAINER, shap_like)
    monkeypatch.setitem(explainers.EXPLAINERS, explainers.FALLBACK_EXPLAINER, fallback)
    chosen, note = explainers.resolve(explainers.DEFAULT_EXPLAINER)
    assert chosen.name == explainers.FALLBACK_EXPLAINER and note

    reports = []
    after = extras.install_extra("analysis", lambda *r: reports.append(r))

    assert after.installed
    stages = [stage for stage, _f, _d in reports]
    for phase in ("resolving", "downloading", "installing", "installed"):
        assert phase in stages, (phase, stages)
    fractions = [f for _s, f, _d in reports]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert any("Successfully installed" in d for *_x, d in reports)
    chosen, note = explainers.resolve(explainers.DEFAULT_EXPLAINER)
    assert (chosen.name, note) == (
        explainers.DEFAULT_EXPLAINER,
        "",
    ), "the running process must see the new package without a restart"


def test_an_editable_install_is_seen_without_a_restart(fake_extra, monkeypatch):
    # A regular install is found by Python's own finder (it notices the new directory); a
    # `.pth`-based one is not, because `.pth` files are read at startup. That is the case the
    # re-read after an install exists for, so it is the case that can show the re-read works.
    module, target = fake_extra
    monkeypatch.setattr(sys, "path", list(sys.path))  # the re-read appends to it
    monkeypatch.setattr(installed, "_site_dirs", lambda: [str(target)])
    monkeypatch.setattr(
        extras,
        "install_argv",
        lambda pkgs: _fake_installer(target, module, editable=True),
    )
    extras.install_extra("analysis")
    assert installed.importlib.util.find_spec(module) is not None


def test_a_failed_install_carries_the_installers_output(fake_extra, monkeypatch):
    module, target = fake_extra
    monkeypatch.setattr(
        extras, "install_argv", lambda pkgs: _fake_installer(target, module, fail=True)
    )
    with pytest.raises(extras.InstallError, match="No matching distribution found"):
        extras.install_extra("analysis")


def test_an_installer_that_succeeds_without_installing_is_not_believed(
    fake_extra, monkeypatch
):
    module, _target = fake_extra
    monkeypatch.setattr(
        extras, "install_argv", lambda pkgs: [sys.executable, "-c", "print('done')"]
    )
    with pytest.raises(extras.InstallError, match="still missing"):
        extras.install_extra("analysis")


def test_cancel_kills_a_quiet_installer(fake_extra, monkeypatch):
    psutil = pytest.importorskip("psutil")
    module, target = fake_extra
    monkeypatch.setattr(
        extras, "install_argv", lambda pkgs: _fake_installer(target, module, hang=True)
    )
    started = time.monotonic()

    def progress(stage, fraction, detail):
        if (target / "pid").exists():  # the installer is up and has gone quiet
            raise OperationCancelled(stage)

    with pytest.raises(OperationCancelled):
        extras.install_extra("analysis", progress)
    assert time.monotonic() - started < 20, "a quiet installer still hears Cancel"
    pid = int((target / "pid").read_text(encoding="utf-8-sig"))
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == "zombie"


def test_nothing_runs_when_nothing_is_missing(monkeypatch):
    monkeypatch.setattr(
        extras, "status", lambda extra: extras.ExtraStatus(extra, ("shap",), ())
    )
    monkeypatch.setattr(
        extras, "run_installer", lambda *a, **k: pytest.fail("ran an installer")
    )
    reports = []
    assert extras.install_extra("analysis", lambda *r: reports.append(r)).installed
    assert reports[-1][:2] == ("installed", 1.0)


def test_a_version_requirement_is_checked_not_just_presence(monkeypatch):
    monkeypatch.setattr(extras.importlib.metadata, "version", lambda name: "0.59.0")
    assert not extras._satisfied("numba>=0.60")
    monkeypatch.setattr(extras.importlib.metadata, "version", lambda name: "0.61.0")
    assert extras._satisfied("numba>=0.60")
