"""No test writes into the repository tree.

**The class: a test that plants a fixture in the source tree and takes it away again.** A guard's
negative case wanted a module that no generator renders, so it wrote
``detail/analysis/zz_orphan_docs.py``, ran its scan and deleted the file in a ``finally``. Alone
that is tidy. In a parallel run it is not: every other test that walks ``detail/`` — and many
guards do (``rglob("*.py")`` then ``read_text``) — can list the file while it exists and open it
after it is gone. Three full runs on three branches failed one test each with
``FileNotFoundError: .../detail/analysis/zz_orphan_docs.py``, each in a different, unrelated scan
(``test_no_orphan_functions``, ``test_user_file_readers``, ``test_no_implicit_network``), and each
passed alone.

**How it hid.** Nothing about the writer fails: it cleans up after itself, so the tree is the same
before and after the run, and a check that compares the two sees nothing. The failure lands in a
different test, in a different file, on a different worker, and only when the timing lines up, so
the run log shows a scattering of one-off failures with nothing in common but a path.

**The rule.** A test writes under ``tmp_path``. A fake module or package is built there, and the
code under test is pointed at it (``_docs_modules(tmp_path)`` in
``tests/test_docs_generated_pages.py``; ``_package_copy`` in ``tests/gui/test_requirements.py``
copies a package out before planting in it). The only names under the repository root a run may
write are the tool-owned ones in ``source_tree.ALLOWED_PARTS``.

**The guard** is ``tests/source_tree.py``, installed for every test process by
``tests/conftest.py``. An audit hook sees each write before it happens, from any test, fixture or
thread, and refuses one into the tree — so the write never lands and no other test sees it — and a
test whose write was refused fails at teardown even if it swallowed the exception. A scan of test
source could not do this: the path is built at runtime (``ANALYSIS / "zz_orphan_docs.py"``), and
the writes the sweep for this change looked for hide behind helpers and fixtures. This file holds
the guard's negative cases. Not covered: writes from a child process and from C code that does not
call Python's file functions.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _source_tree():
    """``tests/source_tree.py``, the one module object ``tests/conftest.py`` installed."""
    name = "detail_test_source_tree"
    if name not in sys.modules:
        path = pathlib.Path(__file__).parent / "source_tree.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


source_tree = _source_tree()


def test_the_guard_would_catch_the_orphan_module_write():
    """The write that caused the flakes, made for real: refused before the file exists."""
    orphan = REPO / "detail" / "analysis" / "zz_orphan_docs.py"
    with pytest.raises(source_tree.SourceTreeWriteError, match="zz_orphan_docs.py"):
        orphan.write_text('"""planted"""\n', encoding="utf-8")
    assert not orphan.exists(), "the write was reported but still landed"
    assert source_tree.take() == [("open", str(orphan))]


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda p: p.touch(), id="touch"),
        pytest.param(lambda p: p.mkdir(), id="mkdir"),
        pytest.param(lambda p: open(p, "a", encoding="utf-8"), id="append"),
        pytest.param(lambda p: os.open(p, os.O_CREAT | os.O_WRONLY), id="os.open"),
        pytest.param(lambda p: os.rename(p.with_name("zz_absent"), p), id="rename"),
    ],
)
def test_the_guard_would_catch_every_spelling_of_a_write(write):
    target = REPO / "detail" / "zz_planted"
    with pytest.raises(source_tree.SourceTreeWriteError):
        write(target)
    assert not target.exists()
    assert source_tree.take(), "refused, but not recorded for the teardown check"


def test_the_guard_would_catch_a_write_the_test_swallowed():
    """Code under test that catches everything still leaves the record the teardown reads."""
    try:
        (REPO / "tests" / "zz_planted.py").write_text("", encoding="utf-8")
    except Exception:  # noqa: BLE001 — the swallowing is the case under test
        pass
    assert source_tree.take() == [("open", str(REPO / "tests" / "zz_planted.py"))]


def test_reads_and_tmp_path_writes_pass(tmp_path):
    (REPO / "detail" / "__init__.py").read_text(encoding="utf-8-sig")
    (tmp_path / "zz_orphan_docs.py").write_text("", encoding="utf-8")
    assert source_tree.take() == []


def test_the_predicates_read_each_event_shape():
    """The pure halves, handed literals in the shapes CPython raises them."""
    written = source_tree.written_paths
    assert written("open", ("/r/a.py", "w", os.O_WRONLY | os.O_CREAT)) == ["/r/a.py"]
    assert written("open", ("/r/a.py", "r", os.O_RDONLY)) == []
    assert written("open", ("/r/a.py", None, os.O_RDONLY)) == []
    assert written("open", (3, "w", os.O_WRONLY)) == []
    assert written("os.rename", ("/r/a", "/r/b", -1, -1)) == ["/r/a", "/r/b"]
    assert written("os.symlink", ("/elsewhere", "/r/link", -1)) == ["/r/link"]
    # shutil.rmtree removes children relative to a directory descriptor; its own event names
    # the top directory, so the relative names are not resolved against the working directory.
    assert written("os.remove", ("a.py", 7)) == []
    assert written("shutil.rmtree", ("/r/pkg", None)) == ["/r/pkg"]
    assert written("sys._getframe", (None,)) == []

    root = pathlib.Path("/r")
    assert source_tree.in_tree("/r/detail/a.py", root)
    assert source_tree.in_tree("/r/new_top_level_file", root)
    assert not source_tree.in_tree("/rx/detail/a.py", root)
    assert not source_tree.in_tree("/tmp/pytest-of-u/a.py", root)
    assert not source_tree.in_tree("/r/detail/__pycache__/a.cpython-312.pyc", root)
    assert not source_tree.in_tree("/r/.git/detail-test-runs.jsonl", root)
    assert not source_tree.in_tree("/r/.pytest_cache/v/cache/lastfailed", root)
    assert not source_tree.in_tree("/r/pytest-cache-files-k2j4/README.md", root)
    assert not source_tree.in_tree("/r", root)
