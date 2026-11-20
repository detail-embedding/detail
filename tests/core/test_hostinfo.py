"""``hostinfo`` — the machine block of the performance report (#394, OQ-RUN-9), and the System
section of ``detail doctor`` (the tests after ``collect``'s).

``psutil`` is a core dependency now, so RAM is a real measurement on a normal install; these
tests pin the three things that made it worth a guard: ``psutil`` is *declared* core, a present
``psutil`` yields a number, and an absent one (a broken install, since it should never be
missing) still degrades to ``UNKNOWN`` rather than raising — the report's job is to survive a
broken machine, not describe one perfectly.

The declaration is its own test because the value tests cannot see it. ``ultralytics`` (the
``yolo`` extra) requires ``psutil``, so on any box with that extra installed — every dev box and
the gate — deleting ``psutil`` from ``[project].dependencies`` leaves it importable and the value
tests green. They catch the regression only in a genuinely plain install, which nothing here
builds.
"""

from __future__ import annotations

import builtins
import pathlib
import re
import sys
import tomllib

from detail.core import hostinfo

REPO = pathlib.Path(__file__).resolve().parents[2]


def _core_dependency_names(pyproject_text: str) -> set[str]:
    """The distribution names in ``[project].dependencies``, with pins, extras and markers cut."""
    dependencies = tomllib.loads(pyproject_text)["project"]["dependencies"]
    return {re.match(r"[A-Za-z0-9._-]+", dep).group(0).lower() for dep in dependencies}


def test_psutil_is_declared_a_core_dependency():
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8-sig")
    assert "psutil" in _core_dependency_names(pyproject)


def test_the_declaration_check_would_catch_psutil_only_in_an_extra():
    # The shape before #394: present on a box with the sam3 extra, absent on a plain install.
    before = """
[project]
dependencies = ["numpy>=2", "pyyaml"]
[project.optional-dependencies]
sam3 = ["einops", "decord", "psutil", "iopath", "timm", "ftfy"]
"""
    assert "psutil" not in _core_dependency_names(before)


def test_the_declaration_check_reads_a_pinned_or_marked_psutil():
    for spelling in (
        "psutil",
        "psutil>=5.9",
        "PSUtil",
        "psutil ; sys_platform != 'emscripten'",
    ):
        text = f'[project]\ndependencies = ["numpy", "{spelling}"]\n'
        assert "psutil" in _core_dependency_names(text), spelling


def test_ram_gb_reads_a_real_number_when_psutil_is_present():
    # No mocking here on purpose: psutil is a core dependency, so this is what every plain
    # install actually sees — the exact gap the "unknown"-forever version of this test hid.
    ram = hostinfo._ram_gb()
    assert ram != hostinfo.UNKNOWN
    assert isinstance(ram, (int, float))
    assert ram > 0


def test_ram_gb_degrades_to_unknown_when_psutil_cannot_be_imported(monkeypatch):
    real_import = builtins.__import__

    def _no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    assert hostinfo._ram_gb() == hostinfo.UNKNOWN


def test_collect_reports_ram_alongside_the_rest_of_the_machine():
    facts = hostinfo.collect()
    assert facts["ram_gb"] != hostinfo.UNKNOWN
    assert facts["ram_gb"] > 0


def test_the_facts_round_trip_through_safe_yaml():
    """The facts are written to YAML and read back with the safe loader — by the report, and by
    anyone comparing machines. torch's `TorchVersion` is a str subclass that YAML wrote as a
    Python object tag, which the safe loader refuses, so a run's recorded machine read back as
    missing and the report fell back to the exporting computer."""
    import yaml

    facts = hostinfo.collect()
    assert yaml.safe_load(yaml.dump(facts, sort_keys=False)) == facts


# --- `detail doctor`'s System section ----------------------------------------------------
#
# The bug-report form asks for `detail doctor` pasted whole, so these rows are what a fixer on
# another machine gets instead of asking. Before them, doctor printed install checks only: no
# build, OS, Python, GPU name, driver or CUDA, so every report needed a second round trip.

_NVIDIA_SMI_HEADER = """\
Wed Nov 19 09:12:44 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
|-----------------------------------------+------------------------+----------------------+
"""

SYSTEM_ROWS = [
    "Build",
    "OS",
    "Python",
    "CPU",
    "RAM",
    "GPU",
    "NVIDIA driver",
    "PyTorch",
    "Packages",
]


def test_nvidia_smi_header_gives_the_driver_and_the_newest_cuda_it_supports():
    assert hostinfo.parse_nvidia_smi(_NVIDIA_SMI_HEADER) == ("580.82.07", "13.0")


def test_nvidia_smi_output_without_a_header_gives_neither():
    assert hostinfo.parse_nvidia_smi("NVIDIA-SMI has failed") == (None, None)


def test_the_system_report_has_every_row_a_bug_report_needs_in_order():
    assert [label for label, _ in hostinfo.system_report()] == SYSTEM_ROWS


def test_no_system_row_carries_a_path_or_the_home_folder():
    """The rows are pasted into public issues. A row naming the environment's folder would
    publish the reporter's user name; the Python row says which *kind* of environment instead.
    """
    text = "\n".join(value for _, value in hostinfo.system_report())
    for leak in (str(pathlib.Path.home()), sys.prefix, "/Users/", "/home/", ":\\"):
        assert leak not in text, leak


def test_package_versions_are_read_from_detail_metadata_without_tooling_extras(
    monkeypatch,
):
    declared = [
        "numpy>=2",
        'pytest>=8; extra == "dev"',
        'ultralytics<8.5,>=8.4; extra == "yolo"',
        'pyobjc-framework-Cocoa; sys_platform == "darwin" and extra == "gui"',
        "mike; extra == 'docs'",
        'numpy<3; extra == "analysis"',  # declared twice: reported once
    ]
    monkeypatch.setattr(hostinfo.metadata, "requires", lambda name: declared)
    monkeypatch.setattr(hostinfo.metadata, "version", lambda name: "1.0")
    names = [name for name, _ in hostinfo.package_versions()]
    assert names == ["numpy", "ultralytics", "pyobjc-framework-Cocoa", "sam3"]


def test_a_declared_package_that_is_not_installed_is_left_out(monkeypatch):
    def _version(name):
        if name == "ultralytics":
            raise hostinfo.metadata.PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr(
        hostinfo.metadata, "requires", lambda name: ["numpy", "ultralytics"]
    )
    monkeypatch.setattr(hostinfo.metadata, "version", _version)
    assert hostinfo.package_versions() == [("numpy", "1.0"), ("sam3", "1.0")]


def test_the_real_install_reports_its_core_packages_and_not_its_test_runner():
    names = {name for name, _ in hostinfo.package_versions()}
    assert {"numpy", "zarr", "click"} <= names
    assert "pytest" not in names  # installed here, but only by the dev extra


def test_the_system_report_survives_a_torch_that_will_not_import(monkeypatch):
    real_import = builtins.__import__

    def _no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    rows = dict(hostinfo.system_report())
    assert rows["PyTorch"] == "not installed, or fails to import"
    assert rows["GPU"] == "none that PyTorch can use"
