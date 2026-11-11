"""No test may leave a ``detail`` module existing twice in the process.

**The class.** A test that wants a cold import deletes ``detail.*`` entries from ``sys.modules``,
re-imports, and puts the old entries back. Re-importing a submodule also rebinds it as an attribute
of its parent package, and restoring ``sys.modules`` does not undo that. From then on the process
holds two copies: ``sys.modules`` and every ``from detail.x.y import Z`` see the old one, while
anything that walks attributes — ``import detail.x.y as m``, ``from detail.x import y``, and every
string-target ``monkeypatch.setattr("detail.x.y.name", ...)`` — reaches the new one.

**How it hid.** ``test_embedding``'s cold-registry test did exactly this to ``detail.models``. The
patch ``"detail.models.sam3.is_importable"`` then landed on a copy ``SAM3`` never reads, so on a
box without the sam3 backend two SAM 3 tests failed — in 12 and 7 of 86 full runs, never alone,
because ``--dist loadfile`` only sometimes put ``test_embedding.py`` earlier on the same worker. On
a box *with* the backend they passed with a patch that reached nothing, and a third guard
(``test_model_availability``'s ``hf_access.readiness`` patch) was silently switched off the same
way. The polluter now gets its cold state from a fresh interpreter.

**The guard.** ``tests/conftest.py::_no_split_modules`` (autouse) compares every ``detail.*`` entry
in ``sys.modules`` with what its parent package holds, before and after each test, and fails the
test during which a new split appeared — naming the polluter instead of a later victim — after
repairing it so the rest of the run is unaffected. This module shows the check sees a split.
"""

from __future__ import annotations

import importlib
import sys

from tests.conftest import _split_detail_modules

# A leaf nothing else in the run patches; re-importing it costs nothing and touches no backend.
_LEAF = "detail.gui.lifetime"


def test_the_check_sees_a_module_restored_behind_its_package():
    importlib.import_module(_LEAF)
    parent, _, child = _LEAF.rpartition(".")
    old = sys.modules[_LEAF]
    try:
        del sys.modules[_LEAF]
        importlib.import_module(_LEAF)  # rebinds the package attribute to a new copy
        sys.modules[_LEAF] = old  # the restore the polluter did
        assert _LEAF in _split_detail_modules()
    finally:
        sys.modules[_LEAF] = old
        setattr(sys.modules[parent], child, old)
    assert _LEAF not in _split_detail_modules()
