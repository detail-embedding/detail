"""CUDA doctor — probe + pure diagnosis (§1)."""

import pytest

from detail.checks import commands as commands_module
from detail.gui import cuda_doctor
from detail.gui.cuda_doctor import ProbeResult, diagnose, probe


@pytest.fixture(autouse=True)
def _uv_present(monkeypatch):
    """Same reasoning as tests/checks/test_commands.py: these fix_command assertions
    are written against a machine that has uv (docs/getting-started.md's own
    documented prerequisite), never whatever the box running the suite happens to
    have. The "no uv" fallback gets its own tests below."""
    monkeypatch.setattr(commands_module.shutil, "which", lambda name: "/usr/bin/uv")


def _p(**kw):
    base = dict(
        cuda=False,
        mps=False,
        cuda_devices=[],
        torch_installed=True,
        torch_cuda_build=None,
        driver_version=None,
        driver_cuda=None,
    )
    base.update(kw)
    return ProbeResult(**base)


def test_cuda_working_is_ready_with_no_fix():
    d = diagnose(
        _p(
            cuda=True,
            cuda_devices=["RTX 4070"],
            torch_cuda_build="12.8",
            driver_version="570.0",
            driver_cuda="12.8",
        )
    )
    assert d.level == "ready" and d.compatible
    assert "RTX 4070" in d.headline
    assert d.fix_command is None and d.problem is None


def test_cpu_only_wheel_on_nvidia_box_suggests_reinstall():
    d = diagnose(_p(driver_version="560.0", driver_cuda="12.6", torch_cuda_build=None))
    assert d.compatible  # an NVIDIA GPU is present → panel active
    assert "CPU-only build" in d.problem
    assert d.fix_command and "cu126" in d.fix_command  # matched to the driver's CUDA
    assert d.fix_command.startswith("uv pip install --reinstall-package")


def test_torch_cuda_newer_than_driver_says_update_driver():
    d = diagnose(
        _p(driver_version="520.0", driver_cuda="11.8", torch_cuda_build="12.8")
    )
    assert "newer than your driver" in d.problem
    assert (
        d.fix_command and "cu118" in d.fix_command
    )  # the older wheel the driver supports


def test_no_nvidia_mps_is_not_compatible():
    d = diagnose(_p(mps=True))
    assert d.headline == "Apple GPU (MPS) detected"
    assert not d.compatible  # greyed on a no-NVIDIA machine
    assert d.fix_command is None


def test_no_torch_says_install():
    d = diagnose(_p(torch_installed=False))
    assert "isn't installed" in d.headline
    assert d.fix_command == "uv pip install -e '.[torch]'"


def test_no_torch_says_install_falls_back_to_plain_pip_without_uv(monkeypatch):
    monkeypatch.setattr(commands_module.shutil, "which", lambda name: None)
    d = diagnose(_p(torch_installed=False))
    assert d.fix_command == "pip install -e '.[torch]'"


def test_reinstall_command_falls_back_to_force_reinstall_without_uv(monkeypatch):
    """uv's --reinstall-package has no plain-pip equivalent flag; --force-reinstall
    reaches the same result (found live: a copy-pasted `uv pip install
    --reinstall-package ...` failed `uv: command not found` on a venv without uv)."""
    monkeypatch.setattr(commands_module.shutil, "which", lambda name: None)
    d = diagnose(_p(driver_version="560.0", driver_cuda="12.6", torch_cuda_build=None))
    assert d.fix_command == (
        "pip install --force-reinstall torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu126"
    )
    assert "--reinstall-package" not in d.fix_command
    assert not d.fix_command.startswith("uv ")


def test_wheel_selection_picks_the_newest_published_tag_at_or_below_the_driver():
    # The driver's CUDA is a maximum, and PyTorch publishes only some versions, so the
    # suggestion is not the driver's version with the dot dropped (#506).
    assert cuda_doctor._cuda_wheel_for("13.2") == "cu132"
    assert cuda_doctor._cuda_wheel_for("13.1") == "cu130"  # no cu131
    assert cuda_doctor._cuda_wheel_for("12.9") == "cu129"
    assert cuda_doctor._cuda_wheel_for("12.8") == "cu128"
    assert cuda_doctor._cuda_wheel_for("12.7") == "cu126"  # no cu127
    assert cuda_doctor._cuda_wheel_for("12.5") == "cu124"  # highest <= 12.5
    assert cuda_doctor._cuda_wheel_for("11.0") is None  # below the oldest we ship
    assert cuda_doctor._cuda_wheel_for(None) is None


def test_a_cuda_13_driver_is_not_offered_a_cuda_12_build():
    # The tester's report: a driver reporting CUDA 13.x was told to install cu128,
    # because the list stopped there.
    d = diagnose(_p(driver_version="595.95", driver_cuda="13.2", torch_cuda_build=None))
    assert d.fix_command.endswith("--index-url https://download.pytorch.org/whl/cu132")


def test_the_suggestion_shows_its_reasoning_in_the_users_numbers():
    d = diagnose(_p(driver_version="560.0", driver_cuda="12.7", torch_cuda_build=None))
    assert d.fix_reason == (
        "Your driver supports up to CUDA 12.7, so the newest PyTorch build it can run "
        "is cu126."
    )


def test_a_driver_older_than_every_build_is_told_to_update():
    d = diagnose(_p(driver_version="450.0", driver_cuda="11.0", torch_cuda_build=None))
    assert d.fix_command is None
    assert d.fix_reason == (
        "Your driver supports up to CUDA 11.0, older than the oldest PyTorch build "
        "listed here (cu118). Update your NVIDIA driver first."
    )


def test_no_reasoning_where_there_is_no_driver_to_reason_from():
    assert diagnose(_p(mps=True)).fix_reason is None
    assert diagnose(_p(torch_installed=False)).fix_reason is None
    assert diagnose(_p(driver_version="560.0", driver_cuda=None)).fix_reason is None


def test_probe_returns_typed_result():
    # real call on whatever the host is — assert shape/types, not host values
    r = probe()
    assert isinstance(r, ProbeResult)
    assert isinstance(r.cuda, bool) and isinstance(r.cuda_devices, list)
    assert r.torch_cuda_build is None or isinstance(r.torch_cuda_build, str)
    assert r.driver_version is None or isinstance(r.driver_version, str)
