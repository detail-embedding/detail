"""What a run is made of, and why it is rows rather than a log.

**The bug.** A run page had one bar for the whole run and a scrolling log under it. On a
detection run that was a bar reading 0% for a long time above lines of
``<time> · <recording> (1/2) · 1% · <eta> left``. Every fact was present and none was
legible: which recording is going, how far in, what failed, what is still waiting. A log is a
history; it was being asked to be a status display.

**The class.** *A surface that reports an aggregate over parts, without reporting the parts.*
Nothing fails — the number is right, the log is complete, the run finishes — and the reader
cannot answer the question they opened the page for. The tell is a page where the interesting
event (one recording failed) is one line among thousands of identical ones.

The guard is over the **derivation**, not the drawing: which rows exist, in what order, in what
state, is the part that can be silently wrong while the widget renders perfectly.

**The second class (issue #28, finding 13).** *A consumer that maps only some of a status
enum, and lets the rest fall through to a default that means something.* `processing_units`
tested ``complete`` and ``failed``; a ``cancelled`` recording then drew as *running* with an
ETA when the live event named it, and a ``partial`` one as *queued*, so a stopped recording read
as work still to come. Distill mapped ``cancelled`` and missed ``partial`` the same way. Both now
read one table, `run_units.STATUS_STATE`, and the guard below is data-driven over `JobStatus`, so
a status added to the enum fails here until it is given a row state on purpose.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QLabel, QProgressBar

from detail.gui.run_units import (
    DONE,
    FAILED,
    GROUP_ABOVE,
    QUEUED,
    RUNNING,
    Unit,
    group_units,
    processing_units,
)


class _Inner:
    def __init__(self, done, total, message="detecting", eta_s=None):
        self.frames_done = done
        self.frames_total = total
        self.message = message
        self.eta_s = eta_s


class _Progress:
    def __init__(self, current, inner):
        self.current_recording = current
        self.inner = inner


def _record(ids, per=None):
    return {"recording_ids": list(ids), "per_recording": per or {}}


# --- the rows that exist ------------------------------------------------------ #
def test_a_recording_the_coordinator_has_not_reached_is_still_a_row():
    """The half a live event cannot supply, and the reason the plan is read too.

    `per_recording` is written as each recording finishes and the progress event names only the
    one in flight, so a run derived from those two alone has no rows for the work still waiting —
    which is half of what someone watching wants to know. `recording_ids` is the plan.
    """
    units = processing_units(
        _record(["r1", "r2", "r3"], {"r1": {"status": "complete"}}),
        _Progress("r2", _Inner(50, 100)),
    )

    assert [u.state for u in units] == [DONE, RUNNING, QUEUED]
    assert units[2].name == "r3"


def test_the_rows_stay_in_the_order_the_run_will_do_them():
    """Not grouped by state: a failure in the middle belongs where it happened."""
    units = processing_units(
        _record(
            ["r1", "r2", "r3", "r4"],
            {"r1": {"status": "complete"}, "r2": {"status": "failed"}},
        ),
        _Progress("r3", _Inner(3, 10)),
    )

    assert [u.state for u in units] == [DONE, FAILED, RUNNING, QUEUED]


def test_a_page_opened_after_the_run_started_is_complete_without_a_live_event():
    """The ordinary way of opening a run page, and the one a live-event-only derivation loses.

    Everything except the row in flight comes off the record, so a reopened page shows the whole
    run immediately rather than filling in as ticks arrive.
    """
    units = processing_units(
        _record(
            ["r1", "r2", "r3"],
            {"r1": {"status": "complete"}, "r2": {"status": "complete"}},
        ),
        None,
    )

    assert [u.state for u in units] == [DONE, DONE, QUEUED]


def test_a_recording_shows_the_name_a_user_gave_it():
    """§11: the id is the tool's, the name is the user's."""
    units = processing_units(
        _record(["recording_004"]), None, names={"recording_004": "subject_03_day2"}
    )

    assert units[0].name == "subject_03_day2"


# --- what a row says ---------------------------------------------------------- #
def test_a_failed_row_names_the_reason_without_the_exception_class():
    """This row is why the page exists, so it has to read at a glance.


    "ValueError: no arena defined" is noise plus a fact; the fact is the half someone acts on.
    """
    units = processing_units(
        _record(["r1"]),
        None,
    )
    assert units  # sanity: the fixture produced a row at all

    units = processing_units(
        _record(
            ["r1"],
            {"r1": {"status": "failed", "error": "ValueError: no arena defined"}},
        ),
        None,
    )

    assert units[0].state == FAILED
    assert units[0].status == "no arena defined"


def test_a_failed_row_keeps_a_reason_that_is_not_an_exception():
    units = processing_units(
        _record(
            ["r1"], {"r1": {"status": "failed", "error": "worker exited: code 137"}}
        ),
        None,
    )

    assert units[0].status == "worker exited: code 137"


def test_a_running_row_reports_its_own_phase_and_progress():
    units = processing_units(
        _record(["r1"]), _Progress("r1", _Inner(710, 1000, "detecting", eta_s=2280))
    )

    assert units[0].state == RUNNING
    assert units[0].fraction == 0.71
    assert "detecting" in units[0].status
    assert "71%" in units[0].status
    assert "38m" in units[0].status, units[0].status


def test_a_running_row_with_no_frames_yet_does_not_divide_by_zero():
    units = processing_units(
        _record(["r1"]), _Progress("r1", _Inner(0, 0, "loading model"))
    )

    assert units[0].fraction == 0.0
    assert "loading model" in units[0].status


# --- grouping ----------------------------------------------------------------- #
def _many(n_done, n_queued):
    units = [Unit(f"d{i}", DONE, 1.0, f"{i}m") for i in range(n_done)]
    units.append(Unit("boom", FAILED, 0.0, "no arena defined"))
    units.append(Unit("live", RUNNING, 0.47, "detecting · 47%"))
    units += [Unit(f"q{i}", QUEUED, 0.0, "queued") for i in range(n_queued)]
    return units


def test_a_short_run_is_not_grouped():
    """Below the threshold a count row is worse than the list it replaces."""
    units = [Unit(f"r{i}", DONE, 1.0, "") for i in range(3)]

    assert group_units(units) == units


def test_a_long_run_folds_the_reference_rows_and_keeps_the_news():
    """Twenty-one rows of which two are news become four.

    The failure and the live row survive individually; the finished and the waiting become
    counts. Asserted as the whole shape, because "it is shorter" is true of every wrong
    grouping too.
    """
    rows = group_units(_many(7, 12))

    assert [r.state for r in rows] == [DONE, FAILED, RUNNING, QUEUED]
    assert rows[0].name == "7 done"
    assert rows[0].is_group
    assert rows[3].name == "12 queued"
    assert not rows[1].is_group, "a failure was folded into a count"
    assert not rows[2].is_group, "the running unit was folded into a count"


def test_grouping_does_not_reorder_the_run():
    """Collapsing *by state* rather than by consecutive run would move a mid-cohort failure to
    the end, and the order of these rows is the order the run happens in."""
    units = (
        [Unit(f"a{i}", DONE, 1.0, "") for i in range(4)]
        + [Unit("boom", FAILED, 0.0, "why")]
        + [Unit(f"b{i}", DONE, 1.0, "") for i in range(5)]
        + [Unit("live", RUNNING, 0.5, "")]
    )

    rows = group_units(units)

    assert [r.state for r in rows] == [DONE, FAILED, DONE, RUNNING]
    assert rows[0].name == "4 done" and rows[2].name == "5 done"


def test_a_group_keeps_its_members_so_it_can_be_opened():
    rows = group_units(_many(7, 12))

    assert len(rows[0].members) == 7
    assert [m.name for m in rows[0].members] == [f"d{i}" for i in range(7)]


def test_the_group_row_reports_the_spread_not_a_total():
    """ "29m – 35m each" answers "are these consistent"; a sum answers nothing about a list
    whose length is already on screen."""
    rows = group_units(_many(7, 12))

    assert rows[0].status == "0m – 6m"


def test_the_threshold_is_the_count_of_units_not_of_groups():
    """Exactly at the threshold nothing folds; one over, it does."""
    exact = [Unit(f"r{i}", DONE, 1.0, "") for i in range(GROUP_ABOVE)]
    over = [Unit(f"r{i}", DONE, 1.0, "") for i in range(GROUP_ABOVE + 1)]

    assert len(group_units(exact)) == GROUP_ABOVE
    assert len(group_units(over)) == 1


# --- distill: two stages, not one bar split by a label ------------------------ #
def _distill_record(
    label_status="pending",
    train_status="pending",
    *,
    label_id=None,
    train_id=None,
    status="running",
    error=None,
):
    record = {
        "status": status,
        "stages": {
            "label": {"status": label_status, "dataset_id": label_id},
            "train": {"status": train_status, "model_id": train_id},
        },
    }
    if error is not None:
        record["error"] = error
    return record


class _DistillProgress:
    def __init__(self, phase, inner_fraction, message="", eta_s=None):
        self.phase = phase
        self.inner_fraction = inner_fraction
        self.message = message
        self.eta_s = eta_s


def test_a_distill_run_shows_one_row_per_stage_in_run_order():
    from detail.gui.run_units import distill_units

    units = distill_units(_distill_record())

    assert [u.name for u in units] == ["Label frames", "Train model"]
    assert [u.state for u in units] == [QUEUED, QUEUED]


def test_a_stage_not_yet_reached_is_queued():
    from detail.gui.run_units import distill_units

    units = distill_units(_distill_record(train_status="pending"))

    assert units[1].state == QUEUED


def test_a_finished_stage_is_done_while_its_sibling_runs():
    """Two rows, not two phases in a label: labelling is over and training is going, at once."""
    from detail.gui.run_units import distill_units

    units = distill_units(
        _distill_record(
            label_status="complete",
            train_status="running",
            label_id="ds_001",
            train_id="model_001",
        ),
        _DistillProgress("training", 0.32, "epoch 19/60", eta_s=2040),
    )

    assert units[0].state == DONE
    assert units[0].fraction == 1.0
    assert units[1].state == RUNNING


def test_the_running_stage_reports_its_own_fraction_and_message_not_the_runs():
    """The run's bar is stage-weighted (`fit_progress_fraction`); the row is not — it is this
    stage's own percentage, which is what answers "how much of training is left"."""
    from detail.gui.run_units import distill_units

    units = distill_units(
        _distill_record(label_status="complete", train_status="running"),
        _DistillProgress("training", 0.19, "epoch 19/60", eta_s=2040),
    )

    assert units[1].fraction == 0.19
    assert "epoch 19/60" in units[1].status
    assert "34m" in units[1].status, units[1].status


def test_a_running_stage_with_no_live_event_yet_shows_zero():
    """A page opened after the run started has no live event, same as `processing_units`'s
    `test_a_page_opened_after_the_run_started_is_complete_without_a_live_event`."""
    from detail.gui.run_units import distill_units

    units = distill_units(_distill_record(label_status="running"))

    assert units[0].state == RUNNING
    assert units[0].fraction == 0.0
    assert units[0].status == "running"


def test_a_live_event_for_the_other_stage_does_not_move_this_row():
    """At the phase boundary the record can say `train` is running before the coordinator's
    next emit has caught up — the row shows zero rather than borrowing labelling's numbers.
    """
    from detail.gui.run_units import distill_units

    units = distill_units(
        _distill_record(train_status="running"),
        _DistillProgress("labeling", 0.9, "labeling frames"),
    )

    assert units[1].state == RUNNING
    assert units[1].fraction == 0.0
    assert units[1].status == "running"


def test_a_failed_stage_names_the_reason_without_the_exception_class():
    from detail.gui.run_units import distill_units

    units = distill_units(
        _distill_record(
            label_status="failed",
            status="failed",
            error="labeling failed: ValueError: no arena defined",
        )
    )

    assert units[0].state == FAILED
    assert "no arena defined" in units[0].status


def test_a_cancelled_stage_is_also_failed_state():
    """`_set_stage` can record a stage's own `run.status`, which is `cancelled` when a stage's
    job was cancelled independently of the run — not only `failed`."""
    from detail.gui.run_units import distill_units

    units = distill_units(_distill_record(train_status="cancelled", status="cancelled"))

    assert units[1].state == FAILED


# --- the wiring, which the derivation tests cannot see ------------------------ #
# `processing_units` can be perfect while nothing on the page calls it, which is the state this
# widget was in for its first render. These drive the page.


def _run_experiment(tmp_path, n, n_done, *, status="running", failed_at=None):
    from detail.api.experiment import Experiment
    from detail.storage import runs as runs_store

    exp = Experiment.create("cohort", tmp_path)
    ids = [exp.add_recording(f"subject_day2_{i:02d}").id for i in range(1, n + 1)]
    per = {}
    for i, rid in enumerate(ids[:n_done]):
        per[rid] = {"run_id": f"det_{i}", "status": "complete", "detail": f"{29 + i}m"}
    if failed_at is not None:
        per[ids[failed_at]] = {
            "status": "failed",
            "error": "ValueError: no arena defined",
        }
    runs_store.write_run(
        exp.dir,
        "run_051",
        {
            "id": "run_051",
            "module_id": "detection",
            "recording_ids": ids,
            "per_recording": per,
            "config": {},
            "status": status,
            "created_at": "2026-09-11T06:42:00Z",
        },
    )
    return exp, ids


class _LiveInner:
    def __init__(self, done, total, message="detecting", eta_s=1140):
        self.frames_done = done
        self.frames_total = total
        self.message = message
        self.eta_s = eta_s


class _LiveProgress:
    def __init__(self, done, total, current, inner):
        self.recordings_done = done
        self.recordings_total = total
        self.current_recording = current
        self.inner = inner
        self.elapsed_s = 600.0
        self.recordings_at_start = 0


def test_a_running_run_shows_a_row_per_recording(qapp, tmp_path, monkeypatch):
    """The page the complaint was about. Assert the **rows**, not that a widget exists — the
    widget existed throughout the failure, holding nothing."""
    from detail.gui.pages import RunDetailPage
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp, _ = _run_experiment(tmp_path, 3, 1)

    page = RunDetailPage(exp, "run_051")

    assert page._panel.units.isVisible() or page._panel.units.isVisibleTo(page)
    assert len(page._panel._units_drawn()) == 3, page._panel._units_drawn()


def test_the_live_event_moves_the_running_row(qapp, tmp_path, monkeypatch):
    """The structured event has to reach the rows.

    `RunProgressWorker` emitted only a rendered `(permille, status)` pair for its whole life, so
    the page could draw one bar and one sentence and nothing else — which is precisely why the
    surface became a log. A test that only checks the bar moves passes with that unchanged.
    """
    from detail.gui.pages import RunDetailPage
    from detail.gui.run_units import RUNNING
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp, ids = _run_experiment(tmp_path, 3, 1)
    page = RunDetailPage(exp, "run_051")

    page._on_progress(_LiveProgress(1, 3, ids[1], _LiveInner(466, 1000)))

    drawn = page._panel._units_drawn()
    live = [u for u in drawn if u.state == RUNNING]
    assert len(live) == 1, drawn
    assert live[0].fraction == 0.466
    assert "detecting" in live[0].status


@pytest.mark.parametrize("attach", ["at_construction", "start_live"])
def test_the_run_page_hears_the_workers_event(qapp, tmp_path, monkeypatch, attach):
    """Both ways a live run reaches `RunDetailPage` — opened on one, and Retry (`_start_live`).

    The tests above call `_on_progress` themselves, which is how Retry's copy of the wiring went
    unnoticed never connecting `progress`: a retried run's rows sat still under a moving bar.
    """
    from detail.gui.pages import RunDetailPage
    from detail.gui.run_units import RUNNING
    from detail.storage import locations
    from tests.gui.conftest import drain_until

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp, ids = _run_experiment(tmp_path, 3, 1)
    handle = _HeldHandle(_LiveProgress(1, 3, ids[1], _LiveInner(466, 1000)))
    if attach == "at_construction":
        page = RunDetailPage(exp, "run_051", live=handle)
    else:
        page = RunDetailPage(exp, "run_051")
        page._start_live(handle)

    def row_moved():
        return [
            u.fraction for u in page._panel._units_drawn() if u.state == RUNNING
        ] == [0.466]

    try:
        assert drain_until(qapp, row_moved), page._panel._units_drawn()
    finally:
        handle.release.set()
        drain_until(qapp, lambda: not page._worker.isRunning())


def test_a_finished_run_shows_the_table_and_not_the_rows(qapp, tmp_path, monkeypatch):
    """Two surfaces, one claim — so they take turns.

    The Results list carries each recording's produced run id, its results and a way to open
    it; the rows carry progress. Both on screen at once is this package's most-repeated class,
    and it is what the first wiring of this did.
    """
    from detail.gui.pages import RunDetailPage
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp, _ = _run_experiment(tmp_path, 3, 3, status="complete")

    page = RunDetailPage(exp, "run_051")

    assert page._panel._units_drawn() == []
    assert _has_per_recording(page), "a finished run lost its Results list"


def _has_per_recording(page) -> bool:
    """The per-recording list, whatever it is headed — it is the run's Results list now."""
    from detail.gui.pages import RunResults

    return bool(page.findChildren(RunResults))


def test_a_running_run_does_not_also_show_the_table(qapp, tmp_path, monkeypatch):
    """The other half of the pair, and the half that was missing.

    Asserting only that a *finished* run has the table passes with the table shown always —
    which is both surfaces on screen at once, the thing taking turns was for. Either half alone
    is satisfied by the bug.
    """
    from detail.gui.pages import RunDetailPage
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp, _ = _run_experiment(tmp_path, 3, 1)

    page = RunDetailPage(exp, "run_051")

    assert page._panel._units_drawn(), "a running run shows no rows"
    assert not _has_per_recording(
        page
    ), "the rows and the table are both on screen — two surfaces for one claim"


def test_a_running_run_keeps_its_cancel_button(qapp, tmp_path, monkeypatch):
    """The regression the first attempt at that shipped.

    Hiding the table with an early `return` took the rest of `_render` with it — including
    Cancel run, the one control a running run needs. Nothing raised; the button was simply not
    there.
    """
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.pages import RunDetailPage
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp, _ = _run_experiment(tmp_path, 3, 1)

    page = RunDetailPage(exp, "run_051")

    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert "Cancel run" in labels, labels


def test_the_rows_survive_a_redraw_without_stacking(qapp, tmp_path, monkeypatch):
    """Rows are rebuilt about twice a second, and `deleteLater` is deferred.

    The first render of this widget drew two rows of text on top of each other, because widgets
    taken out of the layout stay painted at their old geometry until the event loop gets round
    to deleting them. Assert the count of *visible* row widgets, which is what a person sees.
    """
    from detail.gui.pages import RunDetailPage
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp, ids = _run_experiment(tmp_path, 3, 1)
    page = RunDetailPage(exp, "run_051")
    page.resize(900, 400)
    page.show()
    qapp.processEvents()

    for done, current in ((1, ids[1]), (2, ids[2]), (2, ids[2])):
        page._on_progress(_LiveProgress(done, 3, current, _LiveInner(500, 1000)))
        qapp.processEvents()

    visible = [b for b in page._panel.units.findChildren(QProgressBar) if b.isVisible()]
    assert len(visible) == 3, f"{len(visible)} bars on screen for a 3-recording run"


def test_collapsing_a_group_leaves_no_orphan_rows_on_screen(
    qapp, tmp_path, monkeypatch
):
    """The shrink case, which a fixture of constant size never reaches.

    Rows are reused rather than rebuilt, so the ones a smaller draw does not need have to be
    *hidden* — and a test that only ever redraws the same number of rows passes with that
    hiding deleted. Expanding a count row and collapsing it again is the shrink that actually
    happens.
    """
    from PyQt6.QtWidgets import QProgressBar

    from detail.gui.unit_rows import UnitRows
    from detail.gui.run_units import DONE, Unit

    rows = UnitRows()
    rows.resize(800, 400)
    rows.show()
    qapp.processEvents()
    rows.set_units([Unit(f"r{i}", DONE, 1.0, "1m") for i in range(12)])
    qapp.processEvents()
    collapsed = len([b for b in rows.findChildren(QProgressBar) if b.isVisible()])

    rows._toggle("12 done")
    qapp.processEvents()
    expanded = len([b for b in rows.findChildren(QProgressBar) if b.isVisible()])
    rows._toggle("12 done")
    qapp.processEvents()
    again = len([b for b in rows.findChildren(QProgressBar) if b.isVisible()])

    assert collapsed == 1, f"12 finished recordings drew {collapsed} rows"
    assert expanded == 13, f"opening the group drew {expanded} rows"
    assert again == collapsed, (
        f"{again} bars left on screen after collapsing back to {collapsed} — the rows a "
        "smaller draw does not need are still painted"
    )


def test_the_worker_forwards_the_event_itself(qapp):
    """The signal that makes any of this possible — asserted on what it *emits*.

    The first version of this checked `hasattr(RunProgressWorker, "progress")`, which is true of
    a worker that declares the signal and never sends anything down it. Deleting the emit left
    it green. Drive `run()` and collect what comes out.
    """
    from detail.gui.workers import RunProgressWorker

    class _Handle:
        id = "run_051"

        def progress(self):
            yield _LiveProgress(1, 3, "r2", _LiveInner(500, 1000))

    sent = []
    worker = RunProgressWorker(_Handle(), receiver=None)
    worker.emit_to_ui = lambda signal, *args: sent.append((signal, args))
    worker.run()

    payloads = [args[0] for signal, args in sent if args and hasattr(args[0], "inner")]
    assert payloads, (
        "the worker emitted no aggregate event — a page can then draw one bar and one line, "
        "which is the shape the whole complaint was about"
    )
    assert payloads[0].current_recording == "r2"


def test_the_layout_holds_still_while_the_numbers_change(qapp, tmp_path, monkeypatch):
    """The jitter, and why a progress display is laid out in fixed columns.

    Columns sized to their content resize with it, and the bar column takes the remainder — so a
    status going from `running · 0%` to `detecting · 47% · ~19m00s left` and back moved the bar's
    right edge by 130px, twice a second, for the life of the run. Measured across four
    consecutive updates of one run: 659 → 652 → 529 → 558px. It reads as the bar twitching while
    the run is not moving at all.

    Asserted as "nothing about the geometry differs between updates", because any single frame of
    it looks correct — the bug is only visible in the difference.
    """
    from detail.gui.pages import RunDetailPage
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp, ids = _run_experiment(tmp_path, 3, 1)
    page = RunDetailPage(exp, "run_051")
    page.resize(900, 400)
    page.show()
    qapp.processEvents()

    seen = set()
    for message, frac in (
        ("loading model", 0.0),
        ("detecting", 0.02),
        ("detecting", 0.47),
        ("detecting", 0.71),
    ):
        page._panel.tick(int(frac * 1000), f"subject_day2_02 (1/3) · {message}")
        page._on_progress(
            _LiveProgress(1, 3, ids[1], _LiveInner(int(frac * 1000), 1000, message))
        )
        # Activate every level: the page sizes the panel, the panel sizes the grid, and
        # activating only the outermost measures the inner one at its old width.
        page.layout().activate()
        page._panel.layout().activate()
        page._panel.units.layout().activate()
        qapp.processEvents()
        bar = page._panel.units._rows[0].bar
        seen.add((bar.x(), bar.width(), page._panel.units.height()))

    assert len(seen) == 1, f"the rows moved as the numbers changed: {sorted(seen)}"


def test_a_long_status_does_not_widen_its_column(qapp):
    """The mechanism behind the steady layout: the text elides rather than pushing.

    A fixed column width alone is not enough — a label still reports the whole string through its
    size hint, and a grid will hand it more than the minimum if the row has slack.
    """
    from detail.gui.unit_rows import UnitRows

    rows = UnitRows()
    rows.resize(700, 200)
    rows.show()
    qapp.processEvents()

    def _bar_width(status: str) -> int:
        rows.set_units([Unit("r1", RUNNING, 0.5, status)])
        # Geometry is its *previous* value until the layout runs, and Qt has no reason to run it
        # before the next paint — so without this both reads return the same stale number and
        # the test passes against its own reverted fix. It did, twice.
        rows.layout().activate()
        qapp.processEvents()
        return rows._rows[0].bar.width()

    narrow = _bar_width("detecting")
    wide = _bar_width("detecting · 47% · ~19m00s left · ~4h01m run · and more")

    assert narrow == wide, f"a longer status shrank the bar from {narrow}px to {wide}px"


def test_nothing_below_the_panel_moves_as_the_run_updates(qapp, tmp_path, monkeypatch):
    """The guard I should have written the first time, and did not.

    The report was "this region jitters" with an arrow at the whole area below the rows. I
    guarded `bar.x()`, `bar.width()` and `units.height()` — the three things I had just fixed —
    proved those held, and shipped. The form, the config box and the Cancel button, which are
    what the arrow was pointing at, were measured by nothing.

    So this walks **everything below the panel** and asserts none of it moves. Driven by the ETA
    specifically, since that is what changes on a run that is otherwise standing still: hours to
    minutes to seconds, each a different string length.
    """
    from PyQt6.QtWidgets import QLabel, QPlainTextEdit, QPushButton

    from detail.core.progress import format_duration
    from detail.gui.pages import RunDetailPage
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp, ids = _run_experiment(tmp_path, 1, 0)
    page = RunDetailPage(exp, "run_051")
    page.resize(1000, 900)
    page.show()
    page._panel.set_live(True)
    qapp.processEvents()

    def _below() -> dict:
        out = {}
        for kind in (QLabel, QPushButton, QPlainTextEdit):
            for w in page.findChildren(kind):
                if page._panel.isAncestorOf(w):
                    continue  # the panel is what moves things; measure what it moves
                text = getattr(w, "text", lambda: "")() or w.objectName() or str(id(w))
                out[f"{kind.__name__}:{text[:24]}"] = w.mapTo(
                    page, w.rect().topLeft()
                ).y()
        return out

    seen = []
    for eta in (4620, 4380, 3600, 3540, 600, 90, 45, 9):
        page._panel.tick(30, f"subject_day2_01 (0/1) · ~{format_duration(eta)} left")
        page._on_progress(
            _LiveProgress(0, 1, ids[0], _LiveInner(30, 1000, "detecting", eta_s=eta))
        )
        page.layout().activate()
        qapp.processEvents()
        seen.append(_below())

    moved = {k for k in seen[0] if len({s[k] for s in seen}) > 1}
    assert not moved, f"these moved as the estimate changed: {sorted(moved)}"


def test_the_form_below_a_live_panel_does_not_spread(qapp, tmp_path, monkeypatch):
    """The mechanism, so a regression names itself rather than showing up as "it twitches".

    A 0-factor `addStretch()` only marks the spacer *expanding*, which leaves it competing with
    every Preferred widget above it — so the form takes the slack and lays its single-line rows
    out 60px apart instead of 21. A spread form is a form that *redistributes* whenever anything
    above it changes, which is how a panel updating twice a second reaches the bottom of a page.
    """
    from PyQt6.QtWidgets import QLabel

    from detail.gui.pages import RunDetailPage
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp, _ = _run_experiment(tmp_path, 1, 0)
    page = RunDetailPage(exp, "run_051")
    page.resize(1000, 1100)  # deliberately taller than the content, so there IS slack
    page.show()
    page.layout().activate()
    qapp.processEvents()

    rows = sorted(
        w.mapTo(page, w.rect().topLeft()).y()
        for w in page.findChildren(QLabel)
        if w.text().startswith(("Module:", "Status:", "Created:", "Frames:", "Config:"))
    )
    gaps = [b - a for a, b in zip(rows, rows[1:])]

    assert gaps, "the form rows were not found — the fixture stopped reaching them"
    assert max(gaps) < 40, (
        f"form rows are {gaps} apart on a page with slack; they spread rather than the "
        "trailing stretch absorbing it"
    )


def test_a_row_is_the_same_height_whichever_name_widget_it_shows(qapp):
    """The jitter, found in a screen recording after two harnesses said the layout held still.

    A row holds a `QLabel` (a recording's name) and a `QToolButton` (a count row's expander) in
    the **same grid cell**, one shown at a time. A grid row is as tall as the tallest thing in
    it, and a `QToolButton` wants 28px where a `QLabel` wants 14 — so the row's height depended
    on which of the two was visible, and `_render` passed through a state with *both* shown on
    every redraw. Qt is entitled to lay out and paint in that state. Measured off the recording:
    the whole page below hopped **7px down and back** on every update, about twice a second.

    **Why two rounds of guards missed it.** Both measured the settled state — geometry after the
    redraw finished, when the height had already come back. A transient is invisible to that by
    construction. So this asserts the *invariant*: the row is the same height whichever widget is
    in it, which is true in every state including the one nobody meant to render.
    """
    from detail.gui.unit_rows import UnitRows

    rows = UnitRows()
    rows.resize(900, 200)
    rows.show()
    rows.set_units(
        [Unit("r1", RUNNING, 0.5, "detecting"), Unit("r2", QUEUED, 0.0, "queued")]
    )
    rows.layout().activate()
    qapp.processEvents()
    settled = rows.sizeHint().height()

    # The state `_render` used to pass through: the expander shown on an ordinary row.
    rows._rows[0].button.setVisible(True)
    rows.layout().activate()
    qapp.processEvents()
    transient = rows.sizeHint().height()

    assert transient == settled, (
        f"a row is {transient - settled}px taller with its expander shown, so the page below "
        "hops every time the rows are redrawn"
    )


def test_a_count_row_is_no_taller_than_a_recording_row(qapp):
    """The same invariant where it is actually reached, rather than forced.

    A grouped run genuinely shows both kinds of row at once — counts and named recordings — so
    if their heights differ the list is ragged even when nothing is moving.
    """
    from detail.gui.unit_rows import UnitRows

    rows = UnitRows()
    rows.resize(900, 400)
    rows.show()
    rows.set_units(_many(7, 12))
    rows.layout().activate()
    qapp.processEvents()

    heights = {r.bar.height() for r in rows._rows if r.bar.isVisible()}
    labels = {
        (r.name if r.name.isVisible() else r.button).height()
        for r in rows._rows
        if r.bar.isVisible()
    }

    assert len(heights) == 1, f"bars of different heights in one list: {heights}"
    assert len(labels) == 1, f"count rows and recording rows differ in height: {labels}"


def test_render_never_shows_the_widget_it_is_about_to_hide(qapp):
    """Belt to the braces above: the trigger is gone as well as its consequence.

    Asserted by counting how many times each row's button is made visible during one redraw of a
    list with no groups in it. The answer must be zero; it was once per row.
    """
    from detail.gui.unit_rows import UnitRows

    rows = UnitRows()
    rows.resize(900, 200)
    rows.show()
    rows.set_units([Unit("r1", RUNNING, 0.5, "detecting")])
    qapp.processEvents()

    shown = []
    button = rows._rows[0].button
    original = button.setVisible
    button.setVisible = lambda v: (shown.append(v), original(v))[1]
    try:
        rows.set_units([Unit("r1", RUNNING, 0.6, "detecting")])
        qapp.processEvents()
    finally:
        del button.setVisible

    assert (
        True not in shown
    ), f"the expander was shown {shown.count(True)}x while redrawing a list with no groups"


# --- the distill page's wiring, which the derivation tests cannot see --------- #
# `distill_units` can be perfect while nothing on `DistillRunDetail` calls it — the state
# every other run-detail page was in before #372. These drive the page.


def _distill_experiment(
    tmp_path,
    *,
    label_status="complete",
    train_status="running",
    status="running",
    label_id="ds_001",
    train_id="model_001",
):
    from detail.api.experiment import Experiment
    from detail.storage import distill_runs as distill_store

    exp = Experiment.create("distill_cohort", tmp_path)
    record = {
        "id": "distill_003",
        "name": "d1",
        "status": status,
        "created_at": "2026-09-11T06:42:00Z",
        "config": {},
        "stages": {
            "label": {
                "status": label_status,
                "dataset_id": label_id if label_status == "complete" else None,
            },
            "train": {
                "status": train_status,
                "model_id": train_id if train_status != "pending" else None,
            },
        },
        "resume": False,
    }
    distill_store.write_run(exp.dir, "distill_003", record)
    return exp


def test_a_running_distill_shows_a_row_per_stage(qapp, tmp_path, monkeypatch):
    from detail.gui.pages import DistillRunDetail
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _distill_experiment(tmp_path)

    page = DistillRunDetail(exp, "distill_003")

    drawn = page._panel._units_drawn()
    assert [u.name for u in drawn] == ["Label frames", "Train model"]
    assert drawn[0].state == DONE
    assert drawn[1].state == RUNNING


def test_the_live_event_moves_the_running_stage_row(qapp, tmp_path, monkeypatch):
    """The structured event has to reach the rows — `FitProgressWorker` used to flatten every
    event into a rendered `(permille, status)` pair, the same gap `RunProgressWorker.progress`
    closed for processing runs in #372."""
    from detail.gui.pages import DistillRunDetail
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _distill_experiment(tmp_path)
    page = DistillRunDetail(exp, "distill_003")

    page._on_progress(_DistillProgress("training", 0.32, "epoch 19/60", eta_s=2040))

    drawn = page._panel._units_drawn()
    live = [u for u in drawn if u.state == RUNNING]
    assert len(live) == 1, drawn
    assert live[0].fraction == 0.32
    assert "epoch 19/60" in live[0].status


def test_a_finished_distill_hides_the_stage_rows(qapp, tmp_path, monkeypatch):
    """Two surfaces, one claim, same rule as `RunDetailPage`'s Per-recording table: the
    Label:/Train: fields already say the finished state, so the rows step aside."""
    from detail.gui.pages import DistillRunDetail
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _distill_experiment(
        tmp_path, label_status="complete", train_status="complete", status="complete"
    )

    page = DistillRunDetail(exp, "distill_003")

    assert page._panel._units_drawn() == []


class _HeldHandle:
    """A live run that yields one event and then stays open until the test releases it.

    Open, because a finished stream makes the page's `_on_done` rebuild — `RunDetailPage`'s
    clears the live event — and that can land in the same event-loop pass as the progress
    signal, so a test polling the row would see it reset and fail on timing alone.
    """

    status = "complete"
    error = None
    id = "run_051"

    def __init__(self, event):
        import threading

        self._event = event
        self.release = threading.Event()

    def progress(self):
        yield self._event
        self.release.wait(timeout=10)


def _real_distill_event(phase, inner_fraction, message):
    """The coordinator's own event type, so the page reads the fields it will really get."""
    from detail.core.dataclasses import DistillProgress

    index = 0 if phase == "labeling" else 1
    return DistillProgress(
        phase=phase,
        stage_index=index,
        stage_count=2,
        inner_fraction=inner_fraction,
        overall_fraction=(index + inner_fraction) / 2,
        message=message,
    )


def test_a_live_event_runs_a_stage_the_record_still_calls_pending():
    from detail.gui.run_units import distill_units

    units = distill_units(
        _distill_record("pending", "pending"),
        _real_distill_event("labeling", 0.5, "labeling frames"),
    )

    assert [u.state for u in units] == [RUNNING, QUEUED]
    assert units[0].fraction == 0.5


def test_a_resumed_stage_runs_though_the_record_still_says_cancelled():
    """Resume: the stage carries the last attempt's status until the coordinator relaunches it,
    and the first version of the rule knew only ``pending`` — the resumed training drew as a
    failed row under a moving bar."""
    from detail.gui.run_units import distill_units

    units = distill_units(
        _distill_record("complete", "cancelled"),
        _real_distill_event("training", 0.03, "epoch 3/100"),
    )

    assert [u.state for u in units] == [DONE, RUNNING]
    assert units[1].fraction == 0.03


def test_a_complete_stage_stays_done_when_a_resume_replays_it():
    """ "reused labeled dataset" is a labeling event over a stage that is legitimately complete."""
    from detail.gui.run_units import distill_record_lags
    from detail.gui.run_units import distill_units

    record = _distill_record("complete", "cancelled")
    event = _real_distill_event("labeling", 1.0, "reused labeled dataset")

    assert distill_units(record, event)[0].state == DONE
    assert not distill_record_lags(record, event)


@pytest.mark.parametrize(
    "before, stage, phase, row",
    [
        ({}, "label", "labeling", 0),
        ({"label": "complete"}, "train", "training", 1),
    ],
    ids=["label", "train"],
)
def test_the_stage_row_runs_when_the_phase_arrives_before_the_stage_write(
    qapp, tmp_path, monkeypatch, before, stage, phase, row
):
    """The coordinator's real order at a stage boundary, which the tests above never used.

    It emits the new phase, *then* launches the stage's job, *then* writes ``running`` — for
    labelling, seconds apart, because launching reads every source recording's timestamps. The
    page rebuilds (and caches the record) on the phase change and not again for the whole
    stage, so a rebuild landing in that gap left the row "queued · 0%" under a moving bar.
    Every test above writes the record first; this one delivers the event first.
    """
    from detail.gui.pages import DistillRunDetail
    from detail.gui.workers import fit_progress_fraction
    from detail.storage import distill_runs as distill_store
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _distill_experiment(
        tmp_path,
        label_status=before.get("label", "pending"),
        train_status="pending",
    )
    page = DistillRunDetail(exp, "distill_003")

    def deliver(event):  # what FitProgressWorker.run does per event
        fraction, status = fit_progress_fraction(event)
        page._on_tick(round(fraction * 1000), status)
        page._on_progress(event)

    deliver(_real_distill_event(phase, 0.0, f"starting {phase}"))  # rebuilds, caches
    record = exp.distill_run_record("distill_003")
    record["stages"][stage]["status"] = "running"
    distill_store.write_run(exp.dir, "distill_003", record)
    deliver(_real_distill_event(phase, 0.4, "working"))

    drawn = page._panel._units_drawn()[row]
    assert drawn.state == RUNNING, drawn
    assert drawn.fraction == 0.4, drawn


@pytest.mark.parametrize("attach", ["at_construction", "start_live"])
def test_the_page_hears_the_workers_event(qapp, tmp_path, monkeypatch, attach):
    """The link between `FitProgressWorker.progress` and `_on_progress`, which nothing tested.

    The tests above call `_on_progress` themselves, so deleting both
    `progress.connect(self._on_progress)` lines in `_FitDetailPage` left all of `tests/gui`
    green — while in the app the rows would never move. Hand the page a live run and let its
    own worker deliver. Both attach paths: a page opened on a live run, and one a run was
    started from (`_start_live`, e.g. Resume).
    """
    from detail.gui.pages import DistillRunDetail
    from detail.storage import locations
    from tests.gui.conftest import drain_until

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _distill_experiment(tmp_path)
    handle = _HeldHandle(_real_distill_event("training", 0.32, "epoch 19/60"))
    if attach == "at_construction":
        page = DistillRunDetail(exp, "distill_003", live=handle)
    else:
        page = DistillRunDetail(exp, "distill_003")
        page._start_live(handle)

    def train_row_moved():
        return page._panel._units_drawn()[1].fraction == 0.32

    try:
        assert drain_until(qapp, train_row_moved), page._panel._units_drawn()
    finally:
        handle.release.set()
        drain_until(qapp, lambda: not page._worker.isRunning())


# --------------------------------------------------------------------------- #
# an export's units are targets
# --------------------------------------------------------------------------- #
# The sibling of the recording rows above, and it has to be a sibling rather than a parameter:
# `processing_units` is recording-shaped down to its field names. What the rows carry is the
# truth a *multi-target* export cannot get from its bar — `_slice_progress` gives every target
# an equal share, and `clips` can take thousands of times longer than `instances`, so the
# aggregate is coarse and the rows are where the honesty lives.


def _export_record(status="running", outcomes=(), targets=("clips", "instances")):
    return {
        "status": status,
        "target_ids": list(targets),
        "outcomes": [dict(o) for o in outcomes],
    }


def test_an_export_shows_one_row_per_target_in_order():
    from detail.gui.run_units import export_units

    units = export_units(_export_record(targets=("clips", "instances", "ethogram")))
    assert [u.name for u in units] == ["clips", "instances", "ethogram"]


def test_the_target_after_the_last_recorded_one_is_the_one_running():
    """Derived from the record, so a page reopened tomorrow is right without a live event.

    Without this the rows all read *queued* under a bar that is plainly moving — the same
    "a page opened after the run started is complete" rule the recording rows have above.
    """
    from detail.gui.run_units import export_units

    units = export_units(
        _export_record(
            targets=("clips", "instances", "ethogram"),
            outcomes=[{"target": "clips", "status": "complete", "files": ["a", "b"]}],
        )
    )
    assert [u.state for u in units] == ["done", "running", "queued"]
    assert units[0].status == "2 file(s)"


def test_a_failed_target_row_carries_its_reason():
    """A `partial` export's whole point: which one failed, and why, on the row itself."""
    from detail.gui.run_units import export_units

    units = export_units(
        _export_record(
            status="partial",
            outcomes=[
                {"target": "clips", "status": "failed", "error": "the video moved"},
                {"target": "instances", "status": "complete", "files": ["i.csv"]},
            ],
        )
    )
    assert [u.state for u in units] == ["failed", "done"]
    assert units[0].status == "the video moved"


def test_a_finished_export_has_nothing_running():
    """The negative case: a terminal record must not leave a row spinning forever.

    A cancel stops the loop, so targets it never reached have no outcome at all — they stay
    *queued*, rather than one of them reading as in flight on a run that has ended.
    """
    from detail.gui.run_units import export_units

    units = export_units(
        _export_record(
            status="partial",
            targets=("clips", "instances", "ethogram"),
            outcomes=[{"target": "clips", "status": "complete", "files": []}],
        )
    )
    assert "running" not in [u.state for u in units]
    assert [u.state for u in units] == ["done", "queued", "queued"]


# --------------------------------------------------------------------------- #
# an analysis run's units are phases (§16, issue #393)
# --------------------------------------------------------------------------- #
# The third sibling. Unlike a recording or a target, a phase has no plan in the record at all —
# `AnalysisModule.phases` (the module's own declaration) *is* the plan, so every test below
# passes it in directly rather than building a record for it.

PHASES = (
    "loading embeddings",
    "detecting transitions",
    "comparing bouts",
    "clustering",
)


class _Event:
    def __init__(self, message, done=0, total=0):
        self.message = message
        self.frames_done = done
        self.frames_total = total


def test_no_declared_phases_means_no_rows():
    """A module that has not declared any (or an unknown one) draws nothing rather than
    guessing — the same "hidden rather than emptied" rule `LiveRunPanel.set_units` applies.
    """
    from detail.gui.run_units import analysis_units

    assert analysis_units(_Event("loading embeddings"), ()) == []


def test_with_no_live_event_every_phase_is_queued():
    """The honest answer for a page with nothing running yet: there is no per-phase history in
    the record to fall back on the way `processing_units` falls back to `per_recording`.
    """
    from detail.gui.run_units import QUEUED, analysis_units

    units = analysis_units(None, PHASES)

    assert [u.state for u in units] == [QUEUED] * len(PHASES)
    assert [u.name for u in units] == list(PHASES)


def test_a_message_marks_its_phase_running_and_the_ones_before_it_done():
    from detail.gui.run_units import DONE, QUEUED, RUNNING, analysis_units

    units = analysis_units(_Event("detecting transitions · 30 fps"), PHASES)

    assert [u.state for u in units] == [DONE, RUNNING, QUEUED, QUEUED]
    assert units[1].status == "detecting transitions · 30 fps"


def test_a_phase_is_matched_by_its_words_not_the_literal_phrase():
    """ "comparing bouts" is the declared label; the message that carries it has a count in the
    middle — `f"comparing {n} bouts"` — which is not the phrase as a substring."""
    from detail.gui.run_units import RUNNING, analysis_units

    units = analysis_units(_Event("comparing 4,182 bouts"), PHASES)

    assert units[2].state == RUNNING
    assert units[2].name == "comparing bouts"


def test_the_last_phase_running_leaves_every_other_one_done():
    from detail.gui.run_units import DONE, RUNNING, analysis_units

    units = analysis_units(_Event("clustering"), PHASES)

    assert [u.state for u in units] == [DONE, DONE, DONE, RUNNING]


def test_a_running_phase_reports_its_own_fraction():
    from detail.gui.run_units import RUNNING, analysis_units

    units = analysis_units(_Event("clustering", done=460, total=1000), PHASES)

    assert units[3].state == RUNNING
    assert units[3].fraction == 0.46


def test_a_done_phase_carries_no_invented_duration():
    """Unlike a finished recording (which the record timed), an analysis run records no
    per-phase duration — so a done row says nothing rather than making one up."""
    from detail.gui.run_units import DONE, analysis_units

    units = analysis_units(_Event("clustering"), PHASES)

    assert all(u.status == "" for u in units if u.state == DONE)


def test_a_phase_word_in_the_note_does_not_move_the_rows():
    """query-eval's querying messages carry query_tracks' own ``loading <subject>`` in their
    note; matched anywhere in the message, the page went back to "loading" once per subject.
    """
    from detail.gui.run_units import DONE, QUEUED, RUNNING, analysis_units

    units = analysis_units(
        _Event("querying · Rearing · fold 1/3 · loading rec_001 · 4 matches"),
        ("loading", "querying", "scoring"),
    )

    assert [u.state for u in units] == [DONE, RUNNING, QUEUED]


def test_a_done_message_leaves_every_phase_done():
    """The runner's terminal event lands before the page learns the run is over; reading it as
    no phase at all flashed every row back to queued."""
    from detail.gui.run_units import DONE, analysis_units

    units = analysis_units(_Event("done"), PHASES)

    assert [u.state for u in units] == [DONE] * len(PHASES)


# --- every status draws as what it is (issue #28, finding 13) ----------------- #
from detail.core.dataclasses import JobStatus  # noqa: E402

#: What each status must draw as **when the live event names that very recording** — the
#: case that hid the bug, because the fall-through then said *running*. Only ``running`` (and
#: ``pending``, which the coordinator writes the moment it launches) may follow the live event.
_EXPECTED_STATE = {
    JobStatus.PENDING: RUNNING,
    JobStatus.RUNNING: RUNNING,
    JobStatus.COMPLETE: DONE,
    JobStatus.FAILED: FAILED,
    JobStatus.CANCELLED: FAILED,
    JobStatus.PARTIAL: FAILED,
}


def test_every_job_status_has_a_deliberate_row_state():
    """A new `JobStatus` member fails here until it is mapped, in the table the builders read."""
    from detail.gui.run_units import STATUS_STATE

    assert set(STATUS_STATE) == {s.value for s in JobStatus}
    assert set(_EXPECTED_STATE) == set(JobStatus)


@pytest.mark.parametrize("status", list(JobStatus), ids=lambda s: s.value)
def test_a_recording_row_draws_its_status_even_while_the_live_event_names_it(status):
    progress = _Progress("r1", _Inner(50, 100, eta_s=15))
    (unit,) = processing_units(
        _record(["r1"], {"r1": {"status": status.value}}), progress
    )

    assert unit.state == _EXPECTED_STATE[status]
    if status in (JobStatus.CANCELLED, JobStatus.PARTIAL):
        assert unit.status == status.value  # the word, not "running · ~15s left"


@pytest.mark.parametrize("status", list(JobStatus), ids=lambda s: s.value)
def test_a_recording_row_away_from_the_live_event_is_never_running_unless_pending(
    status,
):
    """Off the live event, an active status is *queued*; an ended one draws as ended."""
    (unit,) = processing_units(_record(["r1"], {"r1": {"status": status.value}}))

    expected = _EXPECTED_STATE[status]
    assert unit.state == (QUEUED if expected == RUNNING else expected)


def test_a_cancelled_or_partial_row_keeps_its_reason_after_the_status():
    cancelled, partial = processing_units(
        _record(
            ["r1", "r2"],
            {
                "r1": {"status": "cancelled"},
                "r2": {"status": "partial", "error": "ValueError: stopped at part 2"},
            },
        )
    )
    assert cancelled.status == "cancelled"
    assert partial.status == "partial · stopped at part 2"


@pytest.mark.parametrize(
    "status",
    [s for s in JobStatus if s is not JobStatus.RUNNING],
    ids=lambda s: s.value,
)
def test_a_distill_stage_draws_its_status_through_the_same_table(status):
    """The distill twin reads the same table: ``partial`` used to fall through to *queued*.
    (``running`` is the one stage status whose row depends on a live event, covered above.)
    """
    from detail.gui.run_units import distill_units

    units = distill_units(_distill_record(train_status=status.value, status="running"))
    expected = _EXPECTED_STATE[status]
    assert units[1].state == (QUEUED if expected == RUNNING else expected)
