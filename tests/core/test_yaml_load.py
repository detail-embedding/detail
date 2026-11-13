"""Every YAML read in ``detail/`` goes through ``core.yaml_load``, which parses with libyaml.

**The class.** ``yaml.safe_load`` tokenizes in pure Python, about 2 ms per kilobyte. Nothing at the
line says so, and on the one small file a test hands it nothing is slow. A ``recording.yaml`` is
35 KB on a real experiment: 64 ms to parse on the WSL box and 97 ms on Windows, so the first
listing of a 72-recording experiment (a cluster's clips, the recordings table, a launch) spent
5-7 s parsing, and a montage preparing its clips on six threads paid ~160 ms a parse while the
threads fought over the interpreter. ``CSafeLoader`` is the same safe constructor fed by libyaml:
8x faster on that file, 6.4x over the 3,311 YAML files of the user's experiments, every result
type-for-type equal.

**The rule.** Read YAML with ``yaml_load.safe_load``. The scan reads the parse tree (the
``test_path_serialization.py`` idiom), so a docstring that names ``yaml.safe_load`` is not a hit;
it catches each PyYAML load entry point called through the module (under any alias) or imported
by name. Writing YAML is not this guard's business.
"""

import ast
import importlib
import pathlib
import textwrap

import pytest
import yaml

import detail
from detail.core import yaml_load

_LOADS = {
    "safe_load",
    "load",
    "full_load",
    "unsafe_load",
    "safe_load_all",
    "load_all",
    "full_load_all",
    "unsafe_load_all",
}
_SEAM = pathlib.Path(detail.__file__).parent / "core" / "yaml_load.py"


def _offenders_in(source: str, label: str) -> list[str]:
    tree = ast.parse(source)
    aliases = {"yaml"}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {a.asname for a in node.names if a.name == "yaml" and a.asname}
        if isinstance(node, ast.ImportFrom) and node.module == "yaml":
            found += [f"{label}:{node.lineno}" for a in node.names if a.name in _LOADS]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _LOADS
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            found.append(f"{label}:{node.lineno}")
    return found


@pytest.mark.parametrize(
    "source",
    [
        "yaml.safe_load(text)",
        "yaml.load(text, Loader=yaml.SafeLoader)",
        "yaml.full_load(text)",
        "yaml.unsafe_load(text)",
        "list(yaml.safe_load_all(text))",
        "yaml.load_all(text, Loader=yaml.CSafeLoader)",
        "import yaml as y\ny.safe_load(text)",
        "from yaml import safe_load",
        "from yaml import load as parse",
        "parse = yaml.safe_load",  # a reference, called later
        "def f():\n    import yaml\n    return yaml.safe_load(open('x'))",
    ],
)
def test_the_scan_catches_every_yaml_load_spelling(source):
    assert _offenders_in(textwrap.dedent(source), "x"), source


@pytest.mark.parametrize(
    "source",
    [
        "yaml_load.safe_load(text)",
        "from detail.core import yaml_load\nyaml_load.safe_load(text)",
        '"""Parsed with ``yaml.safe_load``."""',
        "# yaml.safe_load(text) is 8x slower",
        "yaml.dump(data, f)",
        "except yaml.YAMLError:\n    pass",
        "json.load(f)",
    ],
)
def test_the_scan_leaves_the_seam_writers_and_prose_alone(source):
    source = textwrap.dedent(source)
    if source.startswith("except"):
        source = "try:\n    pass\n" + source
    assert _offenders_in(source, "x") == [], source


def test_no_yaml_is_read_outside_the_seam():
    root = pathlib.Path(detail.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path == _SEAM:
            continue
        label = path.relative_to(root.parent).as_posix()
        offenders += _offenders_in(path.read_text(encoding="utf-8-sig"), label)
    assert offenders == [], (
        "read YAML with detail.core.yaml_load.safe_load, which parses with libyaml "
        f"(see its docstring): {offenders}"
    )


def test_the_seam_parses_with_libyaml():
    """The speed is the point: on a PyYAML with libyaml (every wheel this project installs)
    the seam's loader is the C one. A refactor back to ``SafeLoader`` fails here."""
    assert hasattr(yaml, "CSafeLoader"), "this PyYAML was built without libyaml"
    assert yaml_load.LOADER is yaml.CSafeLoader


def test_the_seam_reads_what_safe_load_reads():
    text = textwrap.dedent("""\
        schema_version: 1
        id: recording_000
        created_at: 2026-10-23T10:00:00Z
        day: 2026-10-23
        ratio: .inf
        octal_looking: 0o17
        flags: [yes, no, ~]
        nested: {a: [1, 2.5, "x"], b: {c: null}}
        """)
    got = yaml_load.safe_load(text)
    want = yaml.safe_load(text)
    assert got == want
    assert [type(v) for v in got.values()] == [type(v) for v in want.values()]


def test_the_seam_refuses_what_safe_load_refuses():
    with pytest.raises(yaml.YAMLError):
        yaml_load.safe_load("a: !!python/tuple [1, 2]\n")
    with pytest.raises(yaml.YAMLError):
        yaml_load.safe_load("\x00\x00\x00")  # a zero-filled (torn) file


def test_without_libyaml_the_seam_falls_back_to_the_same_result(monkeypatch):
    monkeypatch.delattr(yaml, "CSafeLoader")
    try:
        fallback = importlib.reload(yaml_load)
        assert fallback.LOADER is yaml.SafeLoader
        assert fallback.safe_load("a: [1, 2]\n") == {"a": [1, 2]}
    finally:
        monkeypatch.undo()
        importlib.reload(yaml_load)
