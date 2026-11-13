"""``hostinfo`` — the machine block of the performance report (#394, OQ-RUN-9).

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

**The class: a third-party library's value written into a report as-is.** ``torch.__version__``
is a ``TorchVersion`` — a ``str`` subclass, so it prints, compares and renders in the Markdown
table exactly like the plain string it looks like. Only ``yaml.dump`` can tell the difference,
and it writes ``torch: !!python/object/new:torch.torch_version.TorchVersion``: a ``machine.yaml``
that ``yaml.safe_load`` refuses, and that the full loader can open only where torch imports —
a report meant to be pasted to someone on another machine. Seen in a real export (2026-10-17).
Nothing at the write is wrong-looking, so the guard is on the *type* of every field rather than
its value: plain ``str`` / ``int`` / ``float`` exactly, plus a ``safe_load`` round trip. The fake
torch below hands back a ``str`` subclass from every probe, so a field that skips ``str(...)``
fails here on a box without torch too; ``test_run_performance`` reads the written file back.
"""

from __future__ import annotations

import builtins
import pathlib
import re
import subprocess
import sys
import tomllib
import types

import pytest
import yaml

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


_PLAIN = (str, int, float)


class _LibraryStr(str):
    """A str subclass, as ``TorchVersion`` is — indistinguishable from ``str`` except by type."""


def _assert_plain_and_round_trips(facts):
    for key, value in facts.items():
        assert type(value) in _PLAIN, f"{key}={value!r} is a {type(value).__qualname__}"
    assert yaml.safe_load(yaml.dump(facts, sort_keys=False)) == facts


def _fake_torch():
    props = types.SimpleNamespace(total_memory=8 * 2**30)
    return types.SimpleNamespace(
        __version__=_LibraryStr("2.6.0+cu124"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: _LibraryStr("NVIDIA GeForce RTX 2070 SUPER"),
            get_device_properties=lambda index: props,
        ),
    )


def test_every_field_is_a_plain_value_that_safe_load_reads_back():
    # The real probes on this box — on any box with torch, `torch.__version__` is the
    # TorchVersion the export wrote as a python/object tag.
    _assert_plain_and_round_trips(hostinfo.collect())


def test_a_library_str_subclass_from_any_torch_probe_is_coerced(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    facts = hostinfo.collect()
    assert facts["torch"] == "2.6.0+cu124"
    assert facts["gpu"] == "NVIDIA GeForce RTX 2070 SUPER"
    assert facts["gpu_memory_gb"] == 8.0
    _assert_plain_and_round_trips(facts)


def test_the_type_check_rejects_the_value_the_export_wrote():
    """The negative case: the check above must bite on exactly what reached a user's file."""
    with pytest.raises(AssertionError, match="_LibraryStr"):
        _assert_plain_and_round_trips({"torch": _LibraryStr("2.6.0+cu124")})
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load(yaml.dump({"torch": _LibraryStr("2.6.0+cu124")}))


_NVIDIA_SMI_HEADERS = {
    "driver_version": (
        "| NVIDIA-SMI 560.94                 Driver Version: 560.94         "
        "CUDA Version: 12.6     |\n",
        ("560.94", "12.6"),
    ),
    # A 600-series driver renamed both fields (seen on 616.92, 2026-11-04); reading only the
    # old names called that machine's working GPU absent.
    "kmd_version": (
        "| NVIDIA-SMI 616.92                 KMD Version: 616.92        "
        "CUDA UMD Version: 13.4     |\n",
        ("616.92", "13.4"),
    ),
}


@pytest.mark.parametrize("form", sorted(_NVIDIA_SMI_HEADERS))
def test_nvidia_smi_reads_the_driver_and_its_cuda(monkeypatch, form):
    # One reader of nvidia-smi for the CUDA doctor, the setup check and this report.
    header, expected = _NVIDIA_SMI_HEADERS[form]
    monkeypatch.setattr(hostinfo.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        hostinfo.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, header, ""),
    )
    assert hostinfo.nvidia_smi() == expected
    assert hostinfo._driver() == expected[0]


def test_nvidia_smi_absent_or_failing_reads_as_no_gpu(monkeypatch):
    monkeypatch.setattr(hostinfo.shutil, "which", lambda name: None)
    assert hostinfo.nvidia_smi() == (None, None)
    assert hostinfo._driver() == hostinfo.UNKNOWN
    monkeypatch.setattr(hostinfo.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        hostinfo.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 9, "", "no devices"),
    )
    assert hostinfo.nvidia_smi() == (None, None)
