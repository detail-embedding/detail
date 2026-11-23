"""Experiment + Distill overview dashboards (§3.1) and the navigable-table click cue."""

from detail.api.experiment import Experiment
from detail.gui import sections as gui_sections
from detail.gui import pages
from detail.gui.qt import QApplication, QLabel, QPushButton, Qt
from detail.storage import locations
from detail.storage import runs as runs_store


def _seed_run(experiment, run_id, status):
    runs_store.write_run(
        experiment.dir,
        run_id,
        {
            "id": run_id,
            "module_id": "detection",
            "recording_ids": ["recording_001"],
            "per_recording": {},
            "config": {},
            "status": status,
            "created_at": "2026-08-01T10:32:00Z",
        },
    )


def test_experiment_summary_counts(qapp, tmp_path):
    exp = Experiment.create("p", tmp_path)
    zero = pages._experiment_summary(exp)  # empty experiment
    assert zero["recordings"].startswith("0 ")
    assert zero["runs"] == "0 run(s)"
    assert "clusterers" not in zero and "benchmarks" not in zero  # removed sections

    exp.create_recording("recording_001")
    exp.create_recording("recording_002")
    _seed_run(exp, "run_001", "complete")
    _seed_run(exp, "run_002", "running")
    _seed_run(exp, "run_003", "failed")
    summary = pages._experiment_summary(exp)
    assert summary["recordings"].startswith("2 ")
    assert (
        "3 run(s)" in summary["runs"]
        and "1 running" in summary["runs"]
        and "1 failed" in summary["runs"]
    )


def test_every_section_has_a_count(qapp, tmp_path):
    """A section the summary has no entry for keeps its "…" forever — `_render_summary` fills
    only the kinds it is given, so a new section without a count fails nothing and simply
    never finishes loading. Datasets and Models arrived in the rail with this obligation.
    """
    exp = Experiment.create("p", tmp_path)
    summary = pages._experiment_summary(exp)
    missing = [s.kind for s in gui_sections.sections() if s.kind not in summary]
    assert not missing, f"the overview shows no count for {missing}"
    assert summary["datasets"] == "0" and summary["models"] == "0"


def test_experiment_overview_builds_and_opens_sections(qapp, tmp_path):
    exp = Experiment.create("p", tmp_path)
    opened = []
    page = pages.ExperimentOverviewPage(exp, on_open_section=opened.append)
    # The sections this build surfaces (§25), read from the declaration rather than repeated
    # here: a literal list would be a fourth copy of the thing `gui/sections.py` exists to
    # keep singular, and it drifted before (the rail said "Processing pipeline" where this
    # page said "Runs"). Data is no longer a section — its content sits at the top.
    shown = [s.kind for s in gui_sections.sections()]
    assert set(page._counts) == set(shown)
    assert "benchmarks" not in shown  # removed (§18)
    buttons = [b for b in page.findChildren(QPushButton) if b.text() == "Open →"]
    assert len(buttons) == len(shown)
    buttons[shown.index("runs")].click()
    assert opened[-1] == "runs"


def test_overview_absorbs_the_data_tab(qapp, tmp_path, monkeypatch):
    # The former Data tab (identity + Export/Save + Delete) now lives on the overview.
    exp = Experiment.create("p", tmp_path)
    deleted = []
    page = pages.ExperimentOverviewPage(
        exp, on_open_section=lambda k: None, on_deleted=lambda: deleted.append(True)
    )
    assert page._name.text() == "p"  # editable identity moved up
    labels = {b.text() for b in page.findChildren(QPushButton)}
    assert {"Export experiment…", "Save", "Delete experiment"} <= labels
    monkeypatch.setattr(
        "detail.gui.pages.QMessageBox.question",
        lambda *a, **k: pages.QMessageBox.StandardButton.Yes,
    )
    page._delete_experiment()
    assert deleted == [True]


def test_experiment_root_shows_overview_not_recordings(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = Experiment.create("p", tmp_path)
    from detail.gui.app import MainWindow

    window = MainWindow()
    window._open_experiment(
        proj
    )  # entering a project shows its overview, not Recordings
    current = window._content.currentWidget()
    assert isinstance(current, pages.ExperimentOverviewPage)
    assert not isinstance(current, pages.ExperimentRecordingsPage)


def test_navigable_table_shows_the_click_cue(qapp):
    nav = pages._table(["A", "B"], [("1", "2")], navigable=True)
    plain = pages._table(["A", "B"], [("1", "2")])
    assert nav.model().columnCount() == 3  # trailing "›" column
    assert nav.model().index(0, 2).data() == "›"
    assert nav.viewport().cursor().shape() == Qt.CursorShape.PointingHandCursor
    assert plain.model().columnCount() == 2  # no cue on a non-navigable table
    assert plain.viewport().cursor().shape() != Qt.CursorShape.PointingHandCursor


def test_table_fits_its_content_height(qapp):
    # An index/list table caps its height to its rows (no Expanding blank space below).
    short = pages._table(["A", "B"], [("1", "2")])
    tall = pages._table(["A", "B"], [(str(i), str(i)) for i in range(50)])
    assert short.maximumHeight() < tall.maximumHeight()
    # cap ≈ header + rows·rowH + frame (a couple px slack for the frame border)
    header = short.horizontalHeader().sizeHint().height()
    row_h = short.verticalHeader().defaultSectionSize()
    assert short.maximumHeight() <= header + 1 * row_h + 4


def _laid_out(page, size=(1400, 900)):
    """Show ``page`` at a real size — an unshown widget reports pre-layout geometry."""
    page.resize(*size)
    page.show()
    QApplication.processEvents()
    return page


def _x(page, widget) -> int:
    return widget.mapTo(page, widget.rect().topLeft()).x()


def _overview(experiment):
    return _laid_out(pages.ExperimentOverviewPage(experiment, lambda _k: None))


def test_the_form_labels_start_at_the_page_margin(qapp, tmp_path, monkeypatch):
    """`QFormLayout` brings its own contents margin, so the one block of the page a user
    reads first sat 9px right of the heading above it and every row below it. Nothing in
    this suite measured an *x* before, which is why a promise `clamp_width`'s docstring
    makes — "everything else on these pages is left-aligned" — went unchecked."""
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    page = _overview(Experiment.create("p", tmp_path))
    heading = next(
        l for l in page.findChildren(QLabel) if l.text() == "Experiment overview"
    )
    labels = [
        l
        for l in page.findChildren(QLabel)
        if l.text() in ("Name:", "Description:", "Folder:")
    ]
    assert len(labels) == 3, [l.text() for l in labels]
    off = [(l.text(), _x(page, l)) for l in labels if _x(page, l) != _x(page, heading)]
    assert (
        not off
    ), f"form labels are off the page's column at {_x(page, heading)}: {off}"
    page.hide()


def test_save_sits_beside_export_not_at_the_far_edge(qapp, tmp_path, monkeypatch):
    """The row is clamped to `CONTENT_WIDTH` with a stretch in it, so Save rendered 760px
    from Export on a 1440px window. `_section_row` states the rule this broke, one screen
    above: the slack goes *after* the controls, never between them."""
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    page = _overview(Experiment.create("p", tmp_path))
    buttons = {b.text(): b for b in page.findChildren(QPushButton)}
    export, save = page._export_btn, buttons["Save"]
    gap = _x(page, save) - (_x(page, export) + export.width())
    assert 0 <= gap <= export.width(), (
        f"Save is {gap}px from Export; they are one row of actions and stop reading as one "
        f"object when the page's slack is put between them"
    )
    page.hide()


def test_a_dashboards_caption_is_as_wide_as_the_rows_it_heads(
    qapp, tmp_path, monkeypatch
):
    """A `_band_caption` draws a full-width rule. Added unclamped above `clamp_width`-ed
    rows it overran them by 318px, so the line under "Sections" pointed past everything it
    captioned. (A second dashboard, the Distill overview, was checked here too until it was
    deleted as unreachable.)"""
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    page = _overview(experiment)
    layout = page.layout()
    widths = [
        layout.itemAt(i).widget().width()
        for i in range(layout.count())
        if layout.itemAt(i).widget() is not None
        and layout.itemAt(i).widget().isVisible()
    ]
    from detail.gui.sizing import CONTENT_WIDTH

    over = [w for w in widths if w > CONTENT_WIDTH]
    # A bare heading label may span the page — it draws no rule and no row. Anything that
    # does is clamped to the one content column.
    assert len(over) <= 1, (
        f"{type(page).__name__} stacks items wider than the {CONTENT_WIDTH}px content "
        f"column its rows use: {over}"
    )
    page.hide()


def _ends_with_stretch(widget) -> bool:
    layout = widget.layout()
    last = layout.itemAt(layout.count() - 1)
    return last is not None and last.spacerItem() is not None


def test_index_pages_top_justify_with_a_trailing_stretch(qapp, tmp_path, monkeypatch):
    # A capped index table (no fill child) must sit under a trailing stretch so the page
    # top-justifies instead of spreading — the fallout of _fit_table_height (gui_design §3).
    from detail.gui.comparison import ComparisonsPage
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    experiment = Experiment.create("p", tmp_path)
    assert _ends_with_stretch(ComparisonsPage(experiment, lambda *a: None))


def test_experiments_page_fills_instead_of_top_justifying(qapp, tmp_path, monkeypatch):
    # The named exception (gui_design §3): a page whose primary content is a fill element
    # expands instead of packing under a trailing stretch. The Experiments list scrolls,
    # so its scroll area must take the slack — with a trailing stretch instead, the cards
    # would bunch at the top and the list would stop scrolling at the window edge.
    from detail.gui.qt import QScrollArea

    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    Experiment.create("p", tmp_path)
    page = pages.HomePage(lambda *a: None)
    assert not _ends_with_stretch(page)
    layout = page.layout()
    fill = [
        i
        for i in range(layout.count())
        if isinstance(layout.itemAt(i).widget(), QScrollArea) and layout.stretch(i) > 0
    ]
    assert fill, "the card list must be the page's fill element"


def test_selection_only_table_has_no_click_cue(qapp):
    # GlobalModelsPage's table is selection-only (row-select enables buttons), so it must
    # NOT carry the navigable cue — the cue would mislead.
    page = pages.GlobalModelsPage()
    assert page._table.viewport().cursor().shape() != Qt.CursorShape.PointingHandCursor
    assert page._table.model().columnCount() == len(page._COLUMNS)  # no trailing "›"
