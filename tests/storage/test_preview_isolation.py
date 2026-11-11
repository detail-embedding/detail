"""A video preview is display-only: no analysing package reads it (§3 *Video preview cache*).

**The class.** The preview cache holds a shrunk, 15 fps copy of every part and masks at that
size. It is machine-local, may be missing or cleared at any moment, and its frames are a
subset of the source's. A stage that read it would analyse different pixels on different
machines, and silently fewer frames — the second video §4 says does not exist. Nothing fails
at the line where such an import is written, and every behavioural test would still pass on a
machine whose cache happens to be built, so a source scan is the only thing that can see it.

Read from the parse tree (the ``test_path_serialization.py`` idiom), so a docstring that names
the module is not an import and an import that wraps onto two lines still is. Every spelling
that reaches the module counts: ``import detail.storage.previews``, ``from detail.storage
import previews``, ``from detail.storage.previews import X``, and the same three for
``detail.api.previews``. The negative case below is the point of the file — a scan whose
pattern quietly stops matching looks exactly like a package that never offends.
"""

import ast
import pathlib
import textwrap

import detail

PREVIEW_MODULES = ("detail.storage.previews", "detail.api.previews")

#: The packages that analyse, train on, or render data for analysis. The four the spec names
#: (pipeline, analysis, trackers, models) plus the rest of the run path: the executor that
#: runs stages, the transforms that render their inputs, and the method / trainer registries.
ANALYSING_PACKAGES = (
    "pipeline",
    "analysis",
    "trackers",
    "models",
    "executor",
    "transforms",
    "query_methods",
    "cluster_methods",
    "trainers",
)


def _imports_a_preview_module(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name in PREVIEW_MODULES or alias.name.startswith(
                tuple(m + "." for m in PREVIEW_MODULES)
            ):
                return alias.name
    if isinstance(node, ast.ImportFrom) and node.module:
        if node.module in PREVIEW_MODULES:
            return node.module
        for alias in node.names:
            if f"{node.module}.{alias.name}" in PREVIEW_MODULES:
                return f"{node.module}.{alias.name}"
    return None


def _offenders(root: pathlib.Path) -> list[str]:
    found = []
    for py in sorted(root.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            hit = _imports_a_preview_module(node)
            if hit:
                found.append(f"{py.relative_to(root).as_posix()}:{node.lineno} {hit}")
    return found


def test_no_analysing_package_imports_the_preview_cache():
    package = pathlib.Path(detail.__file__).parent
    offenders = []
    for name in ANALYSING_PACKAGES:
        folder = package / name
        assert folder.is_dir(), f"{name} moved — point the scan at its new home"
        offenders += [f"{name}/{hit}" for hit in _offenders(folder)]
    assert not offenders, (
        f"analysing code reads the display-only preview cache: {offenders}. A stage reads "
        "the source video; the preview is for looking at (DESIGN_PLAN §3, §4)."
    )


def test_the_scan_catches_every_spelling(tmp_path):
    planted = {
        "a.py": "import detail.storage.previews\n",
        "b.py": "from detail.storage import previews\n",
        "c.py": "from detail.api.previews import PreviewReader\n",
        "d.py": "from detail.api import (\n    clip_montage,\n    previews,\n)\n",
        "e.py": "def f():\n    import detail.api.previews as p\n    return p\n",
    }
    for name, source in planted.items():
        (tmp_path / name).write_text(textwrap.dedent(source), encoding="utf-8")
    assert [hit.split(":")[0] for hit in _offenders(tmp_path)] == sorted(planted)


def test_the_scan_ignores_a_mention_that_is_not_an_import(tmp_path):
    (tmp_path / "ok.py").write_text(
        '"""Never import detail.storage.previews here."""\n'
        "from detail.storage import video\n"
        "from detail.api import clip_montage\n",
        encoding="utf-8",
    )
    assert _offenders(tmp_path) == []
