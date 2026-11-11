"""Every declared section's child enumeration runs, and yields rail-shaped rows.

The rail used to call ``child_items`` for every surfaced section on every rebuild, so a typo
in one of these readers took the window down the moment an experiment was opened and any GUI
test that opened one caught it. Reading them lazily (#316) removes that: a section's reader
now runs only when its folder is expanded, so nothing exercises ``_analysis_runs`` unless a
test happens to click Analysis — and the failure, when it comes, is a folder that quietly
stays empty rather than a crash.

So the coverage moves here, where ``gui/sections.py``'s docstring always said it belonged:
the module is Qt-free on purpose *so that* a test can walk every section without a
QApplication. Data-driven over ``EXPERIMENT_SECTIONS`` rather than ``sections()`` — the
hidden ones (§25) are the ones with the least other coverage — so a section added later is
covered the day it is declared.
"""

import pytest

from detail.api.experiment import Experiment
from detail.gui import sections as gui_sections
from detail.storage import analysis_runs as analysis_store
from detail.storage import runs as runs_store

WITH_CHILDREN = [s for s in gui_sections.EXPERIMENT_SECTIONS if s.children is not None]


@pytest.fixture
def populated(tmp_path):
    """An experiment carrying one of the things the rail's cheap sections list."""
    exp = Experiment.create("p", tmp_path)
    exp.add_recording("mouse_01")
    runs_store.write_run(
        exp.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": [],
            "per_recording": {},
            "config": {},
            "status": "complete",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    directory = analysis_store.analysis_dir(exp.dir, "an_01")
    directory.mkdir(parents=True, exist_ok=True)
    analysis_store.write_analysis_yaml(
        directory,
        {
            "id": "an_01",
            "module_id": "location_export",
            "status": "complete",
            "recording_ids": [],
        },
    )
    return exp


@pytest.mark.parametrize("section", WITH_CHILDREN, ids=lambda s: s.kind)
def test_a_section_enumerates_rail_shaped_rows(section, populated):
    """It runs, and every row is a ``(label, child_id)`` the rail can build a node from.

    A section holding none of its thing returns an empty list — the point is that the reader
    executes, which is what the eager build used to guarantee for free.
    """
    rows = section.child_items(populated)

    assert isinstance(rows, list)
    for row in rows:
        label, child_id = row  # a 2-tuple, or this raises
        assert isinstance(label, str) and label.strip(), f"{section.kind}: blank label"
        assert child_id is not None


@pytest.mark.parametrize("section", WITH_CHILDREN, ids=lambda s: s.kind)
def test_a_section_of_an_empty_experiment_is_empty(section, tmp_path):
    """The state a new experiment is in, and the one the ▸ has to stop claiming."""
    assert section.child_items(Experiment.create("empty", tmp_path)) == []


def test_the_populated_sections_actually_list_what_was_written():
    """Guard the guard: the fixture has to reach the readers, or the sweep above is vacuous.

    A parametrisation that iterates over sections whose lists are all empty asserts nothing
    about the label-building the rail depends on — the shape that has passed elsewhere in this
    repo by never reaching the code it names.
    """
    assert {s.kind for s in WITH_CHILDREN} >= {"recordings", "runs", "analysis"}


@pytest.mark.parametrize("kind", ["recordings", "runs", "analysis"])
def test_a_populated_section_labels_its_rows(kind, populated):
    section = next(s for s in gui_sections.EXPERIMENT_SECTIONS if s.kind == kind)
    rows = section.child_items(populated)

    assert len(rows) == 1, f"{kind} listed {len(rows)} rows for one written record"
    label, _child_id = rows[0]
    expected = {"recordings": "mouse_01", "runs": "run_001", "analysis": "an_01"}[kind]
    assert expected in label
