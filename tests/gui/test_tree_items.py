"""A tree walked by an iterator that outlives the tree corrupts the heap.

**The class.** A Qt helper that is not a ``QObject`` but holds a raw pointer into one, freed after
the thing it points at. ``QTreeWidgetItemIterator`` registers itself with the tree's model when it
is built and unregisters itself in its destructor, without asking whether the model still exists.
So when the only reference to a tree and an iterator over it sit in one frame — a test's locals,
freed in the order they were assigned — the tree goes first and the iterator's destructor writes
into freed memory.

**How it hid.** Nothing fails where it happens. The test body passes; glibc notices the damaged
heap at some *later*, unrelated free and aborts the whole process (``corrupted double-linked
list``, exit 134) — here inside the next fixture's ``processEvents``, so the stack blamed the
GUI teardown drain. No Python exception exists to catch. Under ``-n auto`` the dead worker was
replaced, re-ran the file, died again, and the xdist controller then hung at 0% CPU, so every full
GUI run on ``remote-detail`` hung behind
``test_cluster_workspace.py::test_audit_shows_every_row_after_labelling_on_interpret``. The bug
was older than the merges it was first blamed on: the same test aborted on the commit before them.

**The rule.** Walk a ``QTreeWidget`` with ``lifetime.tree_items``, which builds the iterator and
lets it die inside the call, and returns a list. ``QTreeWidgetItemIterator`` is named nowhere else
in ``detail/`` or ``tests/``. The scan below enforces that, and the subprocess tests pin the
reason: the helper survives a walk beside the tree's only reference, and the raw iterator does not
(on Linux, where glibc detects the write). If that second case ever passes, Qt has fixed the
destructor and this rule — scan included — can be retired.

Swept with this fix: every ``QPainter`` in ``detail/`` paints on ``self`` inside ``paintEvent``
(Qt holds the widget) or calls ``end()`` before its frame ends, and no ``QSignalBlocker``,
``QMutexLocker`` or ``QPersistentModelIndex`` is used, so the iterator is the class's only member
here.
"""

import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
ALLOWED = {"detail/gui/lifetime.py", "tests/gui/test_tree_items.py"}
_ITERATOR = re.compile(r"\bQTreeWidgetItemIterator\b")


def _iterator_sites(root: pathlib.Path) -> list[str]:
    """Every ``path:line`` under ``detail/`` and ``tests/`` that names the raw iterator."""
    sites = []
    for top in ("detail", "tests"):
        for path in sorted((root / top).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if rel in ALLOWED:
                continue
            for number, line in enumerate(
                path.read_text(encoding="utf-8-sig").splitlines(), 1
            ):
                if _ITERATOR.search(line):
                    sites.append(f"{rel}:{number}")
    return sites


def test_no_tree_is_walked_by_a_raw_iterator():
    assert _iterator_sites(REPO) == [], (
        "walk a QTreeWidget with detail.gui.lifetime.tree_items — an iterator that outlives "
        "its tree corrupts the heap"
    )


def test_the_scan_catches_a_planted_iterator(tmp_path):
    planted = tmp_path / "tests" / "gui" / "test_planted.py"
    planted.parent.mkdir(parents=True)
    planted.write_text(
        "from PyQt6 import QtWidgets\n"
        "def walk(tree):\n"
        "    it = QtWidgets.QTreeWidgetItemIterator(tree)\n",
        encoding="utf-8",
    )
    (tmp_path / "detail").mkdir()
    assert _iterator_sites(tmp_path) == ["tests/gui/test_planted.py:3"]


_WALK = """
import gc, sys
from PyQt6.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator
from detail.gui.lifetime import tree_items

app = QApplication([])

def walk_beside_the_only_reference(raw):
    tree = QTreeWidget()
    for i in range(20):
        parent = QTreeWidgetItem(tree, [str(i)])
        for j in range(5):
            QTreeWidgetItem(parent, [str(j)])
    if raw:
        walk = QTreeWidgetItemIterator(tree)
        while walk.value() is not None:
            walk += 1
    else:
        items = tree_items(tree)
        assert len(items) == 120, len(items)

for _ in range(20):
    walk_beside_the_only_reference(sys.argv[1] == "raw")
    gc.collect()
    app.processEvents()
    [QTreeWidget() for _ in range(3)]
"""


def _walk(mode: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": str(REPO),
    }
    return subprocess.run(
        [sys.executable, "-c", _WALK, mode],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_tree_items_survives_a_walk_beside_the_trees_only_reference():
    done = _walk("helper")
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="the freed write is only detected by glibc's allocator checks",
)
def test_the_raw_iterator_still_corrupts_the_heap_negative():
    """The reason for the rule, and the harness shown able to see it: if this starts passing,
    Qt fixed the destructor and the rule above can be retired."""
    done = _walk("raw")
    # Killed by a signal (SIGABRT from glibc, or SIGSEGV), not a Python error in the script.
    assert done.returncode < 0, (
        f"the raw iterator no longer crashes (exit {done.returncode}) — retire this rule: "
        f"{done.stderr[-2000:]}"
    )
