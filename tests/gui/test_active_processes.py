"""The Active-processes page, and the sentence the quit prompt tells about what is in flight.

**The class this closes.** *A prompt that describes an outcome the code no longer produces.* The
quit dialog said one thing — "They keep running in the background if you quit without cancelling
them" — which was true of every family on the day it was written, because every family was a
detached worker process. An **import** is not: it is a thread in this process driven by a queue
that cannot resume (§3), so quitting ends it and the next open marks the recording `failed`.

Nothing could fail over that. The dialog still appeared, still had three buttons, still returned a
choice; only the sentence the user reads it by was wrong, and it was wrong in the direction that
costs them the import. So the sentence is what is asserted here, not that a dialog was built —
and it is read off `RunKind.survives_quit` rather than a check for the import kind, so the next
in-process family is described correctly the day it is added rather than the day someone notices.

**The second class.** *News delivered where nobody is looking.* An import finishing was announced
as "Imported recording_001" for eight seconds and then "Imports complete" for five, along the bottom
edge of the window — and on a local disk that was the only sign one had happened. Both imports in the
report finished inside the second they started, one of them 533 MB, before the user could get
anywhere else; the message said `recording_001` where they had typed `trial_001_escape_vis` (§11);
and the Imports section, which dropped finished rows, could never be looked at in time. Nothing
failed: the message appeared, on schedule, where nobody reads. A finished batch is now announced on
the strip under Export…, where an export's equivalent news already went, by name and with any
failures, and the section keeps recent finished imports. The notice tests below assert the whole
sentence, and the shell test drives the queue's real handlers with a recording named apart from its
id, so printing the id cannot pass.

**The third class.** *An action that reads the cursor where the user made a selection.* The
Archive button asked the table for its **current** row, so with several rows highlighted it
archived one and left the rest, and the table was single-selection besides, so the "several" a user
reached for was never really selectable. Nothing failed: one run vanished from the list and the
others sat there looking unprocessed. The button now reads the selected rows (by ``row_id``, since
sorting moves positions), archives the finished ones in a single re-scan, and leaves a run still
going alone. The tests drive the click with real Ctrl-click input, because the selection mode is
what the person meets, and they fail on the old read, on a single-selection table, and on one
re-scan per run.
"""

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from detail.api import active  # noqa: E402
from detail.api.experiment import Experiment  # noqa: E402
from detail.gui.active_processes import (  # noqa: E402
    _COLUMNS,
    _EXPORT_COLUMNS,
    _IMPORT_COLUMNS,
    ActiveProcessesPage,
)
from detail.gui.app import _quit_message  # noqa: E402
from detail.storage import recording as recording_store  # noqa: E402


def _row(
    kind,
    run_id,
    status="running",
    label="thing",
    experiment_dir=None,
    name="p",
    duration="",
):
    return active.ActiveRun(
        kind=kind,
        experiment_dir=experiment_dir or Path("/tmp/exp"),
        experiment_name=name,
        run_id=run_id,
        label=label,
        status=status,
        created_at="2026-09-13T12:00:00Z",
        duration=duration,
    )


# --------------------------------------------------------------------------- #
# What the quit prompt says
# --------------------------------------------------------------------------- #


def test_runs_alone_are_described_as_surviving_the_quit():
    """The original sentence, kept: a detached worker really does carry on."""
    message = _quit_message({"run": ["run_01", "run_02"]})

    assert "2 run(s)" in message
    assert "keep running in the background" in message


def test_imports_alone_are_described_as_stopping():
    """The sentence that was wrong. Asserted on the *claim* — that they stop and cannot be
    resumed — rather than on any wording, because what makes this a fix is that the user is told
    the outcome they are about to get."""
    message = _quit_message({"import": ["rec_01"]})

    assert "stop when you quit" in message
    assert "cannot be resumed" in message
    assert "keep running in the background" not in message


def test_a_mixed_quit_says_which_ones_stop():
    """The case a single sentence cannot cover, and the reason this is a function rather than an
    f-string at the call site: with both kinds in flight, either blanket sentence is false for
    half of what is running, and the half it is false about is the half that gets lost.
    """
    message = _quit_message({"run": ["run_01"], "import": ["rec_01", "rec_02"]})

    assert "1 run(s) keep going in the background" in message
    assert "2 import(s) stop and cannot be resumed" in message


def test_the_wording_follows_the_table_not_the_import_kind(monkeypatch):
    """Drop `survives_quit` from the import spec and the prompt goes back to promising that an
    import survives. This is what says the sentence is derived rather than special-cased — a
    hardcoded `kind == "import"` would pass every test above and fail this one.
    """
    from dataclasses import replace

    monkeypatch.setattr(
        active,
        "RUN_KINDS",
        tuple(
            replace(spec, survives_quit=True) if spec.kind == "import" else spec
            for spec in active.RUN_KINDS
        ),
    )

    assert "keep running in the background" in _quit_message({"import": ["rec_01"]})


def test_the_leave_running_button_is_not_offered_when_nothing_would_keep_running():
    """A button is the thing a user acts on, so it carries the same promise the body text does.

    With only imports in flight there is nothing to leave running, and a button offering to do
    it is the same false statement as the sentence above it — which is why the dialog's title
    and its destructive button moved off "runs" too, not just its body.
    """
    from detail.gui.app import _anything_survives_quit

    assert _anything_survives_quit({"run": ["run_01"]})
    assert _anything_survives_quit({"run": ["run_01"], "import": ["rec_01"]})
    assert not _anything_survives_quit({"import": ["rec_01"]})
    assert not _anything_survives_quit({})


def _quit_dialog_buttons(monkeypatch, active) -> tuple[str, list[str]]:
    """The title and button labels the quit dialog actually shows, with ``exec`` stubbed out."""
    from PyQt6.QtWidgets import QMessageBox, QWidget

    from detail.gui.app import MainWindow

    shown = {}

    def _capture(box):
        shown["title"] = box.windowTitle()
        shown["buttons"] = [b.text() for b in box.buttons()]
        return 0

    monkeypatch.setattr(QMessageBox, "exec", _capture)
    MainWindow._confirm_quit(QWidget(), active)
    return shown["title"], shown["buttons"]


def test_the_quit_dialog_offers_to_leave_running_only_what_survives(qapp, monkeypatch):
    """The test above pins the predicate; this one pins the dialog that is meant to use it.

    The row this guards says the button carried the same false claim as the sentence and was
    missed twice — and a hardcoded "Leave running" stayed green here, because nothing opened the
    dialog. So read the labels a user acts on, for each shape of what is in flight.
    """
    _, runs = _quit_dialog_buttons(monkeypatch, {"run": ["run_01"]})
    assert any("Leave running" in b for b in runs), runs

    title, imports = _quit_dialog_buttons(monkeypatch, {"import": ["rec_01"]})
    assert not any(
        "Leave running" in b for b in imports
    ), f"with only imports in flight the dialog offers {imports} — nothing will keep running"
    assert "run" not in title.lower(), f"the title {title!r} calls imports runs"


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #


def test_imports_get_their_own_section_rather_than_a_run_row(qapp):
    """An import is not a run, and its row says different things: there is no run id, and the
    two identifying cells are the recording's name and its id. Sections are how this page
    already handles a family whose columns differ (exports), so the check is that an import row
    lands in that section and not in the generic one."""
    page = ActiveProcessesPage(on_open=lambda _r: None)
    page._show(
        [_row("import", "rec_01", label="Habituation day 1"), _row("run", "run_01")]
    )

    imports = page._imports._table.view.model()
    runs = page._runs._table.view.model()
    assert imports.rowCount() == 1, "the import did not reach the Imports section"
    assert runs.rowCount() == 1, "the run did not stay in Runs"
    assert (
        imports.index(0, _IMPORT_COLUMNS.index("recording")).data()
        == "Habituation day 1"
    )
    assert imports.index(0, _IMPORT_COLUMNS.index("id")).data() == "rec_01"


def test_a_remote_run_shows_where_and_cost_in_the_runs_table(qapp):
    """§ run location, duration, cost — the Runs section's own bonus column."""
    page = ActiveProcessesPage(on_open=lambda _r: None)
    remote_row = active.ActiveRun(
        kind="run",
        experiment_dir=Path("/tmp/exp"),
        experiment_name="p",
        run_id="run_01",
        label="detection",
        status="running",
        created_at="2026-09-13T12:00:00Z",
        where="remote: tillicum",
        cost="$1.50",
    )
    page._show([remote_row])

    model = page._runs._table.view.model()
    assert model.index(0, _COLUMNS.index("where")).data() == "remote: tillicum"
    assert model.index(0, _COLUMNS.index("cost")).data() == "$1.50"


def test_export_and_import_sections_show_where_but_not_cost(qapp):
    """Export/import can only ever be local — a cost column there would be a
    column that never has anything in it."""
    assert "cost" not in _EXPORT_COLUMNS
    assert "cost" not in _IMPORT_COLUMNS
    assert "where" in _EXPORT_COLUMNS
    assert "where" in _IMPORT_COLUMNS


def test_duration_is_shown_for_runs_and_exports_but_not_imports(qapp):
    """Runs and exports get `duration_s` written by `run_metrics.completion_fields`; an import
    never does, so its column list omits one rather than always reading blank."""
    assert "duration" in _COLUMNS
    assert "duration" in _EXPORT_COLUMNS
    assert "duration" not in _IMPORT_COLUMNS


def test_a_finished_run_shows_its_duration_in_the_runs_table(qapp):
    page = ActiveProcessesPage(on_open=lambda _r: None)
    page._show([_row("run", "run_01", status="complete", duration="2m05s")])

    model = page._runs._table.view.model()
    assert model.index(0, _COLUMNS.index("duration")).data() == "2m05s"


def test_archive_button_disabled_with_nothing_selected(qapp):
    page = ActiveProcessesPage(on_open=lambda _r: None)
    page._show([_row("run", "run_01", status="complete")])

    assert not page._runs._archive_btn.isEnabled()


def test_archive_button_disabled_for_an_active_row(qapp):
    """Archiving is a dismiss for a *finished* row — a run still going has no reason to be
    hidden from the one place that says so."""
    page = ActiveProcessesPage(on_open=lambda _r: None)
    page._show([_row("run", "run_01", status="running")])
    page._runs._table.view.selectRow(0)

    assert not page._runs._archive_btn.isEnabled()


def test_archive_button_enabled_for_a_finished_row(qapp):
    page = ActiveProcessesPage(on_open=lambda _r: None)
    page._show([_row("run", "run_01", status="complete")])
    page._runs._table.view.selectRow(0)

    assert page._runs._archive_btn.isEnabled()


def test_clicking_archive_dismisses_the_selected_run(qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(active, "archive_run", lambda d, k, r: calls.append((d, k, r)))
    monkeypatch.setattr(ActiveProcessesPage, "refresh", lambda self: None)

    page = ActiveProcessesPage(on_open=lambda _r: None)
    page._show(
        [_row("run", "run_01", status="complete", experiment_dir=Path("/tmp/exp"))]
    )
    page._runs._table.view.selectRow(0)
    page._runs._archive_selected()

    assert calls == [(Path("/tmp/exp"), "run", "run_01")]


def _archiving_page(qapp, monkeypatch, rows):
    """The page over ``rows`` with ``archive_run`` recorded, and the re-scan counted."""
    calls, refreshes = [], []
    monkeypatch.setattr(active, "archive_run", lambda d, k, r: calls.append(r))
    monkeypatch.setattr(
        ActiveProcessesPage, "refresh", lambda self: refreshes.append(1)
    )
    page = ActiveProcessesPage(on_open=lambda _r: None)
    page._show(rows)
    refreshes.clear()  # the page scans once on construction; only the archive's own counts
    return page, calls, refreshes


def _select_rows(section, rows):
    from PyQt6.QtCore import QItemSelectionModel

    view = section._table.view
    model = view.model()
    flags = (
        QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows
    )
    view.clearSelection()
    for row in rows:
        view.selectionModel().select(model.index(row, 0), flags)


def test_archive_dismisses_every_selected_finished_run(qapp, monkeypatch):
    """The report: several rows selected, one archived per click.

    The button read the table's *current* row, so whatever else was highlighted was left behind and
    a batch of finished runs took one click each.
    """
    rows = [
        _row("run", f"run_0{i}", status="complete", experiment_dir=Path(f"/tmp/e{i}"))
        for i in range(1, 5)
    ]
    page, calls, refreshes = _archiving_page(qapp, monkeypatch, rows)
    _select_rows(page._runs, [0, 1, 2])

    page._runs._archive_selected()

    assert sorted(calls) == [
        "run_01",
        "run_02",
        "run_03",
    ]  # not run_04, which was not picked
    assert len(refreshes) == 1  # one re-scan for the batch, not one per run


def test_a_selection_that_includes_a_running_run_archives_only_the_finished(
    qapp, monkeypatch
):
    rows = [
        _row("run", "run_01", status="complete"),
        _row("run", "run_02", status="running"),
        _row("run", "run_03", status="failed"),
    ]
    page, calls, _ = _archiving_page(qapp, monkeypatch, rows)
    _select_rows(page._runs, [0, 1, 2])

    assert (
        page._runs._archive_btn.isEnabled()
    )  # something in the selection can be archived
    page._runs._archive_selected()

    assert sorted(calls) == [
        "run_01",
        "run_03",
    ]  # run_02 is still going and stays listed


def test_a_selection_of_only_running_runs_cannot_be_archived(qapp, monkeypatch):
    rows = [
        _row("run", "run_01", status="running"),
        _row("run", "run_02", status="running"),
    ]
    page, calls, _ = _archiving_page(qapp, monkeypatch, rows)
    _select_rows(page._runs, [0, 1])

    assert not page._runs._archive_btn.isEnabled()
    page._runs._archive_selected()
    assert calls == []


def test_every_section_lets_several_rows_be_selected(qapp):
    """Exports and imports archive from the same button, so they need the same selection."""
    from PyQt6.QtWidgets import QAbstractItemView

    page = ActiveProcessesPage(on_open=lambda _r: None)

    for section in (page._runs, page._exports, page._imports):
        mode = section._table.view.selectionMode()
        assert mode == QAbstractItemView.SelectionMode.ExtendedSelection


def test_ctrl_click_selects_a_second_row_and_archive_takes_both(qapp, monkeypatch):
    """Through real mouse input, not the selection model: the mode is what the user meets."""
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest

    rows = [_row("run", f"run_0{i}", status="complete") for i in range(1, 4)]
    page, calls, _ = _archiving_page(qapp, monkeypatch, rows)
    page.resize(900, 700)
    page.show()
    qapp.processEvents()
    view = page._runs._table.view

    def click(row, modifier):
        y = view.rowViewportPosition(row) + view.rowHeight(row) // 2
        QTest.mouseClick(
            view.viewport(), Qt.MouseButton.LeftButton, modifier, QPoint(30, y)
        )

    click(0, Qt.KeyboardModifier.NoModifier)
    click(2, Qt.KeyboardModifier.ControlModifier)
    page._runs._archive_selected()

    assert sorted(calls) == ["run_01", "run_03"]


def test_a_family_with_no_section_of_its_own_lands_in_runs(qapp):
    """The remainder rule. A kind added to `RUN_KINDS` and forgotten here shows up under the
    generic columns — imprecise, visible, and reportable — rather than in no section at all,
    which is a row that exists in the API, appears on no screen, and fails nothing."""
    page = ActiveProcessesPage(on_open=lambda _r: None)
    page._show([_row("some_future_kind", "x_01")])

    assert page._runs._table.view.model().rowCount() == 1


def test_the_imports_section_says_what_fills_it_when_empty(qapp):
    """A zero-row table still draws its headers, so an empty section without a notice tells
    someone to read rows that are not there. This page's other two sections name where their
    work comes from; so does this one, because nothing is started from this screen."""
    page = ActiveProcessesPage(on_open=lambda _r: None)
    page._show([])

    notice = page._imports._table._notice
    assert notice.isVisible() or not page._imports._table.view.isVisible()
    assert "Add recordings" in notice.text()


def test_double_clicking_an_import_hands_back_that_recording(qapp):
    """The page routes rather than reports: what it owes the shell is the row, carrying the
    recording id the opener needs."""
    opened = []
    page = ActiveProcessesPage(on_open=opened.append)
    page._show([_row("import", "rec_01", label="Habituation day 1")])

    page._imports._open(0)

    assert [run.run_id for run in opened] == ["rec_01"]


def test_opening_a_live_import_row_does_not_kill_the_import(
    qapp, tmp_path, monkeypatch
):
    """Routing to a row's own page must not change the thing the row is about.

    The hazard is one layer down and easy to miss: `_open_active_run` switches experiment
    before it opens anything, and `_activate_experiment` runs `reconcile_imports()`, which
    flips every `queued` / `importing` record to `failed` with no liveness check. So a row
    saying *running*, clicked, could mark the import failed — the listing killing the thing it
    exists to show.

    It does not, because `_ensure_active` returns early when the experiment is already the open
    one, which is the case for any import this app is actually driving. That early return is
    load-bearing rather than incidental, so it is pinned here.

    An import in a *different* experiment is a separate matter and is **not** asserted: by the
    current architecture nothing could be driving it, so the reconcile is right — except in the
    orphaned-queue case, which is #447 and is deliberately out of scope here.
    """
    from detail.gui.app import MainWindow
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    window = MainWindow()
    experiment = Experiment.create("alpha", tmp_path)
    recording = experiment.create_recording("rec_01")
    monkeypatch.setattr(MainWindow, "_open_recording", lambda self, rid: None)
    window._activate_experiment(experiment)

    recording_store.set_import_status(recording.dir, "importing")
    window._open_active_run(
        _row("import", "rec_01", label="rec_01", experiment_dir=experiment.dir)
    )

    assert experiment.get_recording("rec_01").import_status == "importing", (
        "double-clicking a running import marked it failed — the listing changed the state it "
        "was reporting on"
    )


def test_a_finished_import_reads_as_finished_on_the_page(qapp):
    """The row says `complete` and the caption counts it as finished, not running — `_summary`
    counts from what a row is, so a finished import must arrive as `complete`."""
    page = ActiveProcessesPage(on_open=lambda _r: None)
    page._show(
        [_row("import", "rec_01", status="complete", label="trial_001_escape_vis")]
    )

    model = page._imports._table.view.model()
    assert model.rowCount() == 1
    assert model.index(0, _IMPORT_COLUMNS.index("status")).data() == "complete"
    assert (
        page._imports._caption.text()
        == "0 running · 1 finished — double-click a row to open it"
    )


# --------------------------------------------------------------------------- #
# Column widths: the bug this page was reported for
# --------------------------------------------------------------------------- #


def test_every_section_stays_column_resizable_after_a_refresh(qapp):
    """`_Section._size_columns` re-applies its column modes after every fill (`set_runs`),
    to re-hand the stretch column its slack once `IndexTable.set_rows` has resized
    everything to content (see its docstring). It used to do that by looping
    `ResizeToContents` over every *other* column, which silently relocks them: a user
    could drag a border right after the page was built and not at all once a single run
    landed. Driven through `_show`, the real refresh path, on all three sections — the
    stretch column differs per section (`experiment` for Runs, `to` for Exports,
    `recording` for Imports), so a fix that only widened one section's loop would still
    leave the others locked.
    """
    from tests.gui.layout_checks import locked_columns

    page = ActiveProcessesPage(on_open=lambda _r: None)
    page._show(
        [
            _row("run", "run_01"),
            _row("export", "run_02"),
            _row("import", "rec_01", label="Habituation day 1"),
        ]
    )

    for name, section in (
        ("runs", page._runs),
        ("exports", page._exports),
        ("imports", page._imports),
    ):
        locked = locked_columns(section._table.view)
        assert (
            not locked
        ), f"{name} section: column(s) {locked} cannot be dragged after a refresh"


# --------------------------------------------------------------------------- #
# When an import finishes
# --------------------------------------------------------------------------- #
# The second class in the module docstring: news delivered where nobody is looking.


def test_one_finished_import_is_announced_by_name():
    from detail.gui.app import _import_notice_text

    assert _import_notice_text([("trial_001_escape_vis", True, "")]) == (
        "Imported trial_001_escape_vis. See it under Active processes."
    )


def test_a_batch_is_announced_by_count_and_names():
    from detail.gui.app import _import_notice_text

    assert _import_notice_text([("cam_a", True, ""), ("cam_b", True, "")]) == (
        "Imported 2 recordings: cam_a, cam_b. See them under Active processes."
    )


def test_a_long_batch_names_three_and_counts_the_rest():
    """A sixty-recording import is not one sentence of sixty names."""
    from detail.gui.app import _import_notice_text

    results = [(f"cam_{i}", True, "") for i in range(60)]
    assert _import_notice_text(results) == (
        "Imported 60 recordings: cam_0, cam_1, cam_2 and 57 more. "
        "See them under Active processes."
    )


def test_a_failure_is_named_with_its_reason():
    """The status bar used to carry the error for eight seconds and then lose it; the strip
    keeps it until the user navigates."""
    from detail.gui.app import _import_notice_text

    assert _import_notice_text(
        [("cam_a", True, ""), ("cam_b", False, "OSError: disk full")]
    ) == (
        "Imported cam_a. 1 failed: cam_b — OSError: disk full. "
        "See them under Active processes."
    )


def test_an_empty_batch_says_nothing():
    from detail.gui.app import _import_notice_text

    assert _import_notice_text([]) == ""


def test_finishing_an_import_raises_the_notice_by_name_and_clears_the_status_bar(
    qapp, tmp_path, monkeypatch
):
    """Through the shell's real handlers, in the order the queue emits them.

    The recording is named apart from its id so the name/id rule can fail: a fixture whose id and
    name were both `rec_01` would pass with the id printed.
    """
    from pathlib import Path

    from detail.api.imports import ImportProgressFold, RecordingImportSpec
    from detail.gui.app import MainWindow
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    window = MainWindow()
    experiment = Experiment.create("alpha", tmp_path)
    recording = experiment.add_recording("trial_001_escape_vis")
    assert recording.id != recording.name
    window._experiment = experiment
    fold = ImportProgressFold()
    fold.add(
        [
            RecordingImportSpec(
                id=recording.id, name=recording.name, video_parts=[Path("v.mp4")]
            )
        ]
    )

    window._on_import_started(recording.id)
    window._on_import_progress(fold.progress(recording.id, 0.5, "Copying"))
    assert (
        window.statusBar()
        .currentMessage()
        .startswith("Importing trial_001_escape_vis (1/1) — Copying")
    )
    assert not window._import_bar.isHidden(), "no bar while the import ran"
    assert window._import_bar.value() == 500, "the bar dropped the fraction"

    window._on_import_item_done(recording.id, True, "")
    window._on_import_progress(fold.finished(recording.id, True))
    window._on_import_idle()

    assert not window._export_notice.isHidden(), "the finished import was not announced"
    assert window._export_notice._label.text() == (
        "Imported trial_001_escape_vis. See it under Active processes."
    )
    assert (
        window.statusBar().currentMessage() == ""
    ), "the status bar still says the import is in progress after it finished"
    assert window._import_bar.isHidden(), "the bar outlived the import"


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def test_listing_imports_costs_one_read_per_recording(tmp_path, monkeypatch):
    """**Assert the slope, not the total.**

    `active_ids` is called inline on the UI thread in `closeEvent` — a declared exemption in
    `test_ui_thread_blocking.py`, granted when it read one record per *run*. Reading imports
    makes it O(recordings), which is the recorded class: nothing on a local disk, seconds over a
    share. Every `Recording` property is its own `recording.yaml` read and this wants three of
    them, so without the pin the slope is the number of fields, not the number of recordings.

    A wall clock measures nothing here — on this box a stat is tens of microseconds. Count the
    syscalls, and take the slope between two sizes so a per-recording cost cannot hide in the
    constant.

    Measured: **3.0** per recording pinned — one listing probe, one cache-validating stat, one
    parse — against **5.0** with the pin removed, where the two extra fields each cost their own
    round trip. The threshold sits between them rather than on either: at 3.0 it would go red on
    an incidental stat, and at 5.0 it would not notice a field being re-read.
    """
    import collections
    import io
    import os
    from unittest.mock import patch

    from detail.storage import locations

    def round_trips(count):
        root = tmp_path / f"ws{count}"
        root.mkdir()
        monkeypatch.setattr(locations, "experiments_root", lambda: root)
        experiment = Experiment.create("alpha", root)
        for index in range(count):
            recording = experiment.create_recording(f"rec_{index:03d}")
            recording_store.set_import_status(recording.dir, "queued")

        seen = collections.Counter()
        real_stat, real_open, real_scandir = os.stat, io.open, os.scandir

        def spy(func, tag):
            def wrapped(*args, **kwargs):
                seen[tag] += 1
                return func(*args, **kwargs)

            return wrapped

        with (
            patch("os.stat", side_effect=spy(real_stat, "stat")),
            patch("io.open", side_effect=spy(real_open, "open")),
            patch("os.scandir", side_effect=spy(real_scandir, "scandir")),
        ):
            assert len(experiment.import_runs) == count
        return sum(seen.values())

    small, large = 4, 24
    slope = (round_trips(large) - round_trips(small)) / (large - small)

    assert slope <= 3.5, (
        f"{slope:.1f} round trips per recording. The quit guard reads this on the UI thread, "
        f"so a slope that tracks the field count rather than the recording count is a freeze "
        f"on a network workspace — check that the read is still inside `read_cache.pinned()`."
    )
