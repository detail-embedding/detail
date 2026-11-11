"""Fail-closed guard: every zarr store and every path rename in ``detail/`` goes through
the retry seam.

**The class:** *a hazard diagnosed at one call site and mitigated there, rather than at
the seam every call site passes through.* zarr commits each chunk file and each metadata
rewrite with its own ``tmp.replace(dst)``; on a network share that rename-over-existing
intermittently returns a transient ``PermissionError`` (WinError 5/32). ``core.atomic``
has ridden that out for our own writers since the first Windows report, and
``storage/zarr_io.py`` applied it **per call** — one ``retry_on_lock`` around each
``create_array``, each resize, each ``open_group``.

A per-call mitigation is complete on the day it is written and incomplete the moment a
second module writes a zarr. ``storage/frame_view.py`` is that second module: it wrapped
its ``open_group`` — which writes one small ``zarr.json`` — and left the four
``create_array`` calls that write every chunk file bare. A long run on an SMB share died
part-way, after many successful internal renames,
having made **zero** retry attempts on the one that failed. The wrapping was inverted
relative to the risk, and nothing about the code said so.

**The rule:** a zarr group is opened through ``storage/zarr_store.py``, whose store
retries every write zarr makes underneath it, and a rename of a path is
``core.atomic.replace``. Neither is a call an author has to remember to wrap.

Two scans, both on the AST rather than the text so a name in a docstring or a monkeypatch
target written as a string is not a hit:

- any attribute call on the ``zarr`` module (``zarr.open_group``, ``zarr.open``,
  ``zarr.create_array``, …) outside the seam module;
- ``os.replace(...)``, and any single-argument ``.replace(x)`` — which is
  ``Path.replace``, since ``str.replace`` takes two — outside ``core/atomic.py``.

``tests/storage/test_frame_view_transient.py`` is the behavioural half: it injects the
transient and shows a frame-view write rides it out.
"""

import ast
from pathlib import Path

import detail

_PKG = Path(detail.__file__).parent

#: The one module allowed to call zarr's own openers — it is the seam.
_ZARR_SEAM = "storage/zarr_store.py"

#: The one module allowed to rename a path — it is the seam.
_REPLACE_SEAM = "core/atomic.py"


def _calls(source: str):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            yield node


def _is_zarr_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "zarr"
    )


def _is_path_replace(node: ast.Call) -> bool:
    """``os.replace(a, b)`` or ``<something>.replace(x)``.

    The single-argument form is the discriminator that makes this scannable at all:
    ``str.replace`` requires two arguments, so a one-argument ``.replace`` is
    ``Path.replace`` and nothing else. ``dataclasses.replace`` is named out because it is
    a different function that happens to share the word.
    """
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "replace":
        return False
    if isinstance(func.value, ast.Name):
        if func.value.id == "os":
            return True
        if func.value.id == "dataclasses":
            return False
    return len(node.args) == 1 and not node.keywords


def _offenders(predicate, seam: str) -> list[str]:
    found = []
    for py in sorted(_PKG.rglob("*.py")):
        rel = py.relative_to(_PKG).as_posix()
        if rel == seam:
            continue
        source = py.read_text(encoding="utf-8-sig")
        for node in _calls(source):
            if predicate(node):
                found.append(f"detail/{rel}:{node.lineno}: {ast.unparse(node.func)}")
    return found


def test_every_zarr_group_is_opened_through_the_store_seam():
    offenders = _offenders(_is_zarr_call, _ZARR_SEAM)
    assert not offenders, (
        "open zarr groups through detail/storage/zarr_store.py::open_group — its store "
        "retries the transient network-share replace zarr makes internally, which a "
        "retry wrapped around this call site does not reach:\n" + "\n".join(offenders)
    )


def test_every_path_rename_goes_through_atomic_replace():
    offenders = _offenders(_is_path_replace, _REPLACE_SEAM)
    assert not offenders, (
        "rename through detail/core/atomic.py::replace, which rides out the sharing "
        "violation a concurrent reader provokes on Windows:\n" + "\n".join(offenders)
    )


def test_the_scans_would_catch_one():
    """Guard the guards: a source scan that matches nothing passes vacuously forever."""
    zarr_call = next(_calls('root = zarr.open_group(str(p), mode="w")'))
    assert _is_zarr_call(zarr_call)
    assert not _is_zarr_call(next(_calls("root.create_array(name, data=x)")))

    assert _is_path_replace(next(_calls("os.replace(tmp, path)")))
    assert _is_path_replace(next(_calls("tmp.replace(path)")))
    # ...and not on the two-argument string method, nor on the seam's own callers.
    assert not _is_path_replace(next(_calls('name.replace("-", "_")')))
    assert not _is_path_replace(next(_calls("atomic.replace(tmp, path)")))
    assert not _is_path_replace(next(_calls("dataclasses.replace(row, id=1)")))
