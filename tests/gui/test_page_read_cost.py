"""What a page costs in **round trips**, per recording.

The class: *a cost model that assumes local disk, applied to a workspace a user can point at
a network share* — the one `test_dirscan.py` and `test_ui_thread_blocking.py` already record,
measured here on the surface rather than the listing. Its second half, the one this file was
written for: **a cached document re-validated once per field access**. Every ``Recording``
property is its own ``read_recording(self.dir).get(...)``, and a hit still costs a ``stat``,
so the recordings table asked one ``recording.yaml`` for eight different fields and paid nine
round trips a row, seconds of frozen window on a network share. Nothing is wrong at the
line where it is written.

**Assert the slope, not the total.** A total is a magic number that drifts with every column
added and tells you nothing about the shape; the slope is the property — how much this page
costs *per recording* — and it is what turns a workspace into a freeze. Both surfaces here sit
at the floor of **2.0**: one ``dirscan`` probe to list a recording, one read of its record.

A wall clock measures nothing here: on a local disk a stat is so fast that the same
page looks perfect. Count the syscalls.
"""

import os
import threading
from unittest.mock import patch

import pytest

from detail.api.experiment import Experiment
from detail.gui import pages as gui_pages
from detail.gui.qt import QTest
from detail.storage import locations
from detail.storage import read_cache
from detail.storage import recording as recording_store
from tests.gui.conftest import drain_busy_calls, drain_thread_pool
from tests.round_trips import counting

#: The two sizes the slope is taken between. Small enough to stay fast, far enough apart that
#: a per-recording cost cannot hide in the constant.
SIZES = (4, 24)

#: One listing probe + one record read. Anything above this is a field re-read.
FLOOR_PER_RECORDING = 2.0


def _round_trips(qapp, root, build, settle):
    """Round trips ``build`` costs under ``root``, on **every** thread, drained until ``settle``.

    Both threads are counted deliberately: moving a read off the UI thread does not make it
    free, and a guard that counted only the UI thread would call an unchanged page fixed the
    day it grew a worker. ``tests/gui/test_smoke_construct.py`` is where *which thread* is
    asserted; this file is about how many. Only reads under ``root`` — the test's own
    ``tmp_path`` — count, so another test's leftover poller is not charged to the page
    (``tests/round_trips.py``).
    """
    with counting(root) as seen:
        build()
        waited = 0
        while waited < 10_000 and not settle():
            qapp.processEvents()
            QTest.qWait(10)
            waited += 10
        qapp.processEvents()
    return sum(seen.values())


def _quiesce(qapp) -> None:
    """Let every worker finish before the next measurement starts.

    A fill still running from the previous case lands in the next one's count — which reads as
    a page that got worse. Traced once already while writing this file: it reported
    ``MetadataPage`` at 3.0 instead of 2.0. Waits for the work itself, not a fixed number of
    pumps, which a loaded machine can outrun.
    """
    drain_busy_calls(qapp)
    drain_thread_pool(qapp)
    qapp.processEvents()


def _experiment(tmp_path, n: int):
    root = tmp_path / f"n{n}"
    root.mkdir()
    experiment = Experiment.create("p", root)
    for i in range(n):
        experiment.create_recording(f"recording_{i:03d}")
    return experiment


def _recordings_page(experiment):
    return gui_pages.ExperimentRecordingsPage(
        experiment, lambda _r: None, lambda: None, on_import=lambda *a: None
    )


def _slope(qapp, tmp_path, build, settle_for):
    """Round trips per recording, from a build at each of :data:`SIZES`.

    Each size is built **twice** and only the second is counted: the first warms the caches,
    so what is measured is the steady state a user actually clicks through rather than the
    one-off cost of opening the experiment.
    """
    costs = {}
    for n in SIZES:
        experiment = _experiment(tmp_path, n)
        read_cache.clear_read_cache()
        recording_store.clear_recording_cache()
        holder = {}

        def make(h=holder, e=experiment):
            h["page"] = build(e)

        make()
        _quiesce(qapp)
        costs[n] = _round_trips(
            qapp, tmp_path, make, lambda h=holder: settle_for(h["page"], n)
        )
        _quiesce(qapp)
    low, high = SIZES
    return (costs[high] - costs[low]) / (high - low), costs


@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)


def test_the_recordings_table_costs_one_read_per_recording(qapp, tmp_path):
    """It asked the same file for eight fields; now it asks once."""
    slope, costs = _slope(
        qapp,
        tmp_path,
        _recordings_page,
        lambda page, n: page._table.view.model().rowCount() == n,
    )

    assert slope <= FLOOR_PER_RECORDING, (
        f"{slope:.1f} round trips per recording ({costs}) — a row wants several fields of one "
        "record, so read it once: wrap the build in read_cache.pinned()"
    )


def test_the_metadata_table_is_already_at_the_floor(qapp, tmp_path):
    """A page that reads each record once, pinned here so nobody 'optimizes' it.

    It is the control for the test above: the same measurement over a page with no
    amplification, which is what makes 2.0 a floor rather than a number that happened to pass.
    """
    slope, costs = _slope(
        qapp, tmp_path, gui_pages.MetadataPage, lambda _page, _n: True
    )

    assert slope <= FLOOR_PER_RECORDING, f"{slope:.1f} per recording ({costs})"


def test_the_count_would_see_an_unpinned_build(qapp, tmp_path):
    """Guard the guard: a measurement that cannot rise passes vacuously forever.

    ``pinned()`` is neutralized for one build, and the slope must go back over the floor —
    otherwise this file is asserting something the page could not violate.
    """
    import contextlib

    @contextlib.contextmanager
    def unpinned():
        yield

    with patch.object(read_cache, "pinned", unpinned):
        slope, costs = _slope(
            qapp,
            tmp_path,
            _recordings_page,
            lambda page, n: page._table.view.model().rowCount() == n,
        )

    assert slope > FLOOR_PER_RECORDING, (
        f"{slope:.1f} per recording ({costs}) with the pin removed — the measurement cannot "
        "see the shape it exists to catch"
    )


def test_the_recordings_table_reads_nothing_on_the_ui_thread(qapp, tmp_path):
    """Collapsing the reads is not a substitute for moving them (`storage/dirscan.py`).

    One read per recording is still ~1.5 s of frozen window over 60 of them on a share, so
    the rows are built in a worker; this asserts the UI thread pays **nothing** for them.
    """
    experiment = _experiment(tmp_path, SIZES[1])
    read_cache.clear_read_cache()
    recording_store.clear_recording_cache()
    main = threading.main_thread()
    on_ui = []
    real = os.stat

    def spy(*args, **kwargs):
        if threading.current_thread() is main:
            on_ui.append(args[0] if args else None)
        return real(*args, **kwargs)

    with patch("os.stat", side_effect=spy):
        page = _recordings_page(experiment)
        assert (
            not on_ui
        ), f"the table read {len(on_ui)} paths on the UI thread: {on_ui[:3]}"
        waited = 0
        while waited < 10_000 and page._table.view.model().rowCount() != SIZES[1]:
            qapp.processEvents()
            QTest.qWait(10)
            waited += 10

    assert (
        page._table.view.model().rowCount() == SIZES[1]
    )  # ...and the rows still arrive
    _quiesce(qapp)


# --------------------------------------------------------------------------- #
# the same rule on the *launch* surface: a selection handler is O(recordings) too
#
# A launch form reacts to the recording selection, and that reaction reads disk per
# recording. The rule is the same one this file is about, and the mechanism for obeying it
# already exists — `ConfigWidget.resolve_recordings` (heavy, Qt-free, run off the UI thread by
# the launch dialog) plus `apply_recordings` (UI). What was missing is anything that *asks*.
#
# Measured before this: the tracking form cost **13.0** round trips per recording — 780 over
# sixty, synchronously, on every checkbox click — because `spec_groups` walks all four
# processing modules, `arena_name_groups` reads each recording's arenas, and `source_rate` ran
# three separate times. `location_export` cost 5.0. Both are 2.0 and off-thread now.
#
# 2.0 is the floor and not reducible: one `stat` and one `open` to read each selected
# recording's record once. So for the rest the question is not *how much* but *where*, and the
# ceiling below is the enforced remainder.
# --------------------------------------------------------------------------- #
#: Forms whose selection handler still runs on the UI thread, with what it costs per
#: recording. A ceiling, not a floor: migrating one means deleting its line, and this test
#: fails if a form not listed here appears, or if a listed one turns out to be fixed. The
#: number is the measured slope, so a form getting *worse* fails too.
#:
#: Every one of these is a real freeze on a network share, growing with the recordings.
#: They are listed rather than fixed here to keep one reviewable change; the rule is enforced
#: for anything new from today.
FORMS_STILL_ON_THE_UI_THREAD: dict[str, float] = {
    "detection": 2.0,
    "query": 2.0,
}


def _form_slope(qapp, tmp_path, module_id, ui):
    """Round trips per recording for one launch form's selection reaction."""
    from detail.gui.modules.registry import ConfigWidget

    per_size = {}
    root = tmp_path / module_id
    root.mkdir(parents=True, exist_ok=True)  # `_experiment` only creates its own leaf
    for size in SIZES:
        experiment = _experiment(root, size)
        recordings = experiment.recordings
        form = ui.config_widget()
        form.on_experiment_set(experiment)
        # Warm, then measure: the first pass pays for caches the second would be charged for.
        form.apply_recordings(recordings, form.resolve_recordings(recordings))
        _quiesce(qapp)
        recording_store.clear_recording_cache()
        per_size[size] = _round_trips(
            qapp,
            root,
            lambda f=form, r=recordings: f.apply_recordings(r, f.resolve_recordings(r)),
            lambda: True,
        )
        _quiesce(qapp)
    lo, hi = min(SIZES), max(SIZES)
    slope = (per_size[hi] - per_size[lo]) / (hi - lo)
    off_thread = type(form).resolve_recordings is not ConfigWidget.resolve_recordings
    return slope, off_thread


def test_a_launch_form_that_reads_per_recording_resolves_off_the_ui_thread(
    qapp, tmp_path
):
    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES, GUI_MODULES

    registries = {**GUI_MODULES, **ANALYSIS_GUI_MODULES}
    assert len(registries) >= 8, f"the sweep lost the registries: {sorted(registries)}"

    unlisted, fixed, worse = [], [], []
    for module_id, ui in sorted(registries.items()):
        try:
            slope, off_thread = _form_slope(qapp, tmp_path, module_id, ui)
        except Exception as exc:  # pragma: no cover - surfaced, not skipped
            pytest.fail(f"{module_id}'s config form could not be measured: {exc!r}")
        if slope <= 0.0:
            continue  # reads nothing per recording; the rule does not apply
        if off_thread:
            if module_id in FORMS_STILL_ON_THE_UI_THREAD:
                fixed.append(module_id)
            continue
        allowed = FORMS_STILL_ON_THE_UI_THREAD.get(module_id)
        if allowed is None:
            unlisted.append((module_id, slope))
        elif slope > allowed:
            worse.append((module_id, slope, allowed))

    assert not unlisted, (
        "these launch forms read disk per recording on the UI thread and are not declared: "
        f"{unlisted}. Split the reaction into `resolve_recordings` (heavy, Qt-free) + "
        "`apply_recordings` (UI) — `modules/tracking.py` is the worked example."
    )
    assert (
        not worse
    ), f"a declared form got more expensive: {worse}. The entry is a ceiling, not a budget."
    assert not fixed, (
        f"{fixed} now resolve off the UI thread — delete their entries from "
        "FORMS_STILL_ON_THE_UI_THREAD. An allowance that excuses nothing is one nobody "
        "will question."
    )


def test_the_migrated_forms_are_actually_off_the_ui_thread(qapp, tmp_path):
    """The positive half, naming the two this change fixed.

    Without it the test above passes just as well on a package where *every* form is listed,
    which is the state it is meant to be ratcheting away from.
    """
    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES, GUI_MODULES

    registries = {**GUI_MODULES, **ANALYSIS_GUI_MODULES}
    for module_id in ("tracking", "location_export"):
        slope, off_thread = _form_slope(
            qapp, tmp_path, module_id, registries[module_id]
        )
        assert off_thread, f"{module_id} stopped resolving off the UI thread"
        assert slope <= 2.0, (
            f"{module_id} costs {slope} round trips per recording; 2.0 is the floor "
            f"(one stat + one read of each selected record)"
        )
