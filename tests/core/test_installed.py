"""A package probe inside the app sees what a freshly started interpreter would (#506).

**The class.** A check of what is installed, run inside the long-lived app, answers from the
interpreter's startup view. An editable install (``pip install -e ../sam3``) registers through
a ``.pth`` file in site-packages, and Python reads ``.pth`` files only at startup, so
``importlib.util.find_spec`` keeps returning ``None`` for a package installed while the app is
open, even after ``importlib.invalidate_caches()``. Every run worker is a fresh interpreter and
does see it, so the app and its workers disagree. The SAM 3 page (whose check runs in a
subprocess) said "You're set" while Quick segment, which loads the model in the app, could not
import it.

**How it hid.** A regular install needs no help, because the path finder notices a new directory
in site-packages by itself. Every test process starts after its packages are installed, and
nothing in CI installs a package mid-run. Confirmed on Windows on 2026-09-30 with a package laid
out like sam3:

=================================  ======================  =====
probe in the running process       editable (import hook)  plain
=================================  ======================  =====
right after the install            not found               found
after ``invalidate_caches()``      not found               found
in a fresh subprocess              found                   found
after ``site.addsitedir``          found                   found
=================================  ======================  =====

**The rule.** ``detail/`` asks ``core.installed.is_importable``, which re-reads site-packages'
``.pth`` files on a miss, and never calls ``find_spec`` itself. The scan below holds that, and the
behaviour tests plant both ``.pth`` shapes setuptools writes and watch a bare ``find_spec`` miss
what ``is_importable`` finds.

**Where it stops.** A reinstalled *compiled* package (a CPU torch swapped for a CUDA one) cannot
be re-read into a running process at all. The GPU / CUDA page tells the user to restart instead
(``gui.cuda_doctor.RESTART_AFTER_FIX``).
"""

import ast
import importlib
import importlib.util
import pathlib
import site
import sys
import uuid

import pytest

import detail
from detail.core import installed

_PROBES = {"find_spec", "find_loader"}


def _bare_probes(source: str) -> list[int]:
    """Line of every ``find_spec`` / ``find_loader`` call in ``source``, however spelled."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        )
        if name in _PROBES:
            lines.append(node.lineno)
    return lines


def test_detail_probes_packages_only_through_is_importable():
    root = pathlib.Path(detail.__file__).resolve().parent
    seam = pathlib.Path(installed.__file__).resolve()
    offenders = [
        f"{py.relative_to(root.parent).as_posix()}:{line}"
        for py in sorted(root.rglob("*.py"))
        if py.resolve() != seam
        for line in _bare_probes(py.read_text(encoding="utf-8-sig"))
    ]
    assert not offenders, (
        f"bare package probes {offenders}: inside the app these answer from startup and miss "
        "a backend installed editable while it is open. Use "
        "detail.core.installed.is_importable."
    )


@pytest.mark.parametrize(
    "source",
    [
        "import importlib.util\nimportlib.util.find_spec('sam3')\n",
        "from importlib.util import find_spec\nok = find_spec('sam3') is not None\n",
        "from importlib import util\nutil.find_spec(\n    'sam3'\n)\n",
        "import importlib.util as u\nmissing = [m for m in ('a',) if u.find_spec(m) is None]\n",
        "import pkgutil\npkgutil.find_loader('sam3')\n",
    ],
)
def test_the_scan_would_catch_a_bare_probe(source):
    assert _bare_probes(source)


def test_the_scan_ignores_a_probe_named_in_prose():
    source = '"""find_spec(m) misses an editable install."""\n# find_spec("sam3")\n'
    assert not _bare_probes(source)


@pytest.fixture
def site_packages(tmp_path, monkeypatch):
    """An empty site-packages already on ``sys.path``, as startup leaves it.

    Its ``.pth`` files are unprocessed, which is the state of a real one after an install
    the running process did not see. The import system is restored afterwards.
    """
    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_dir)])
    monkeypatch.setattr(site, "ENABLE_USER_SITE", False)
    monkeypatch.setattr(sys, "path", [*sys.path, str(site_dir)])
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sys, "path_importer_cache", dict(sys.path_importer_cache))
    before = set(sys.modules)
    yield site_dir
    for name in set(sys.modules) - before:
        if name.startswith("__editable___probe_"):
            del sys.modules[name]


def _package(root: pathlib.Path) -> tuple[str, pathlib.Path]:
    """A package outside every import path, as an editable install leaves the source."""
    name = f"probe_{uuid.uuid4().hex[:12]}"
    src = root / "src"
    (src / name).mkdir(parents=True)
    (src / name / "__init__.py").write_text("X = 1\n", encoding="utf-8")
    return name, src


def _install_editable_with_import_hook(site_dir: pathlib.Path, src: pathlib.Path, name):
    """setuptools' ``_TopLevelFinder`` shape, which ``pip install -e ../sam3`` writes."""
    finder = f"__editable___{name}_finder"
    init = (src / name / "__init__.py").as_posix()
    (site_dir / f"{finder}.py").write_text(
        "import importlib.util\n"
        "import sys\n"
        "class _Finder:\n"
        "    @classmethod\n"
        "    def find_spec(cls, fullname, path=None, target=None):\n"
        f"        if fullname != {name!r}:\n"
        "            return None\n"
        "        return importlib.util.spec_from_file_location(\n"
        f"            fullname, {init!r}, submodule_search_locations=[]\n"
        "        )\n"
        "def install():\n"
        "    if _Finder not in sys.meta_path:\n"
        "        sys.meta_path.append(_Finder)\n",
        encoding="utf-8",
    )
    (site_dir / f"__editable__.{name}-0.1.0.pth").write_text(
        f"import {finder}; {finder}.install()\n", encoding="utf-8"
    )


def _install_editable_with_path_line(site_dir: pathlib.Path, src: pathlib.Path, name):
    """setuptools' ``_StaticPth`` shape: the ``.pth`` names the source directory."""
    (site_dir / f"__editable__.{name}-0.1.0.pth").write_text(
        f"{src}\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    "install",
    [_install_editable_with_import_hook, _install_editable_with_path_line],
    ids=["import-hook", "path-line"],
)
def test_an_editable_install_made_after_startup_is_importable(
    site_packages, tmp_path, install
):
    name, src = _package(tmp_path)
    install(site_packages, src, name)

    # The fixture reproduces the bug: the running process cannot see the package, and
    # clearing the finder caches does not help.
    assert importlib.util.find_spec(name) is None
    importlib.invalidate_caches()
    assert importlib.util.find_spec(name) is None

    assert installed.is_importable(name) is True
    # The refresh changed what the process imports, not only the answer: a later
    # ``import`` in the same process (Quick segment's model load) succeeds.
    assert importlib.util.find_spec(name) is not None


def test_a_package_that_is_not_installed_still_reads_missing(site_packages):
    assert installed.is_importable(f"probe_{uuid.uuid4().hex[:12]}") is False


def test_repeated_refreshes_leave_the_import_system_as_one_refresh_did(
    site_packages, tmp_path
):
    name, src = _package(tmp_path)
    _install_editable_with_import_hook(site_packages, src, name)
    _install_editable_with_path_line(site_packages, src, name)
    installed.refresh_site_packages()
    path, meta_path = list(sys.path), list(sys.meta_path)
    installed.refresh_site_packages()
    installed.refresh_site_packages()
    assert sys.path == path
    assert sys.meta_path == meta_path
