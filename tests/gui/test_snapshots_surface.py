"""Pinned artifacts on screen (§21).

Snapshots worked and were invisible: pressing **Finish** in the workspace wrote one and nothing on
screen said so, which is a guarantee nobody can see. These cover the two surfaces that fix that —
the run-detail table and the read-only mode it opens — and the one derived fact they rest on.

"Used by" is the interesting assertion. #242 deliberately removed `caused_by` from the snapshot
record, because a fact checkable from the store gets no second writer (§21), so the link from a
snapshot to the comparison that read it is read back off the comparisons. A test that only checked
the column *rendered* would pass against an empty column forever.
"""

import numpy as np

from detail.api.comparison import run_comparison
from detail.api.experiment import Experiment
from detail.gui.snapshots import snapshot_columns, snapshot_rows
from detail.storage import analysis_runs
from detail.storage import behavior_output as bo
from detail.storage import frame_view as fv
from detail.storage import metadata as md
from detail.storage import temporal_runs

N_FRAMES, PERIOD = 100, 10_000_000


def _cohort(tmp_path, run_id="an_01"):
    """An experiment with one authored-looking artifact a comparison can measure."""
    exp = Experiment.create("cohort", tmp_path)
    for recording in ("r1", "r2"):
        exp.create_recording(recording)
        md.upsert_row(exp.dir, recording, {"animal": "A"})
    return _add_run(exp, run_id)


def _add_run(exp, run_id):
    """A second analysis run in an experiment that already exists."""
    axis = (np.arange(N_FRAMES) * PERIOD).astype(np.int64)
    run_dir = analysis_runs.analysis_dir(exp.dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_runs.write_analysis_yaml(
        run_dir,
        {"id": run_id, "module_id": "temporal_clustering", "status": "open"},
    )
    temporal_runs.write_temporal_state(
        run_dir,
        temporal_runs.TemporalClusteringState(
            embed_run_id="emb_01",
            track_id=0,
            method="dtw_finch",
            source_recording_ids=["r1", "r2"],
            granularity=4,
        ),
    )
    instances, coverage = [], []
    for recording in ("r1", "r2"):
        for i in range(3):
            instances.append(
                bo.BehaviorInstance(
                    recording,
                    0,
                    int(axis[i * 3]),
                    int(axis[i * 3 + 1]),
                    i * 3,
                    i * 3 + 1,
                    "c01",
                    "rearing",
                )
            )
        coverage.append(
            bo.CoverageRow(
                recording, 0, 0, N_FRAMES * PERIOD, N_FRAMES * PERIOD, N_FRAMES
            )
        )
    bo.write_behavior_output(
        run_dir,
        instances=instances,
        coverage=coverage,
        manifest={
            "producer": {"module_id": "temporal_clustering", "run_id": run_id},
            "origin": "predicted",
            "label_space": "pooled",
            "vocabulary": ["rearing"],
            "names": {"c01": ["rearing"]},
            "overlap": "exclusive",
            "behavior_types": {"rearing": "state"},
        },
    )
    for recording in ("r1", "r2"):
        fv.write_frame_view(
            run_dir,
            recording_id=recording,
            subject=0,
            t_ns=axis,
            observed=np.ones(N_FRAMES, bool),
            instances=[i for i in instances if i.recording_id == recording],
        )
    return exp


def _run(exp, run_id="an_01"):
    return exp.authored_run(run_id)


def _name(exp, names, run_id="an_01"):
    """Set the names on disk without reprojecting.

    `name_clusters` rebuilds the artifact from the fit, and this fixture has no fit — it writes the
    §19 artifact directly, which is all these rows read.
    """
    run_dir = analysis_runs.analysis_dir(exp.dir, run_id)
    state = temporal_runs.read_temporal_state(run_dir)
    state.names.update({str(k): str(v) for k, v in names.items()})
    temporal_runs.write_temporal_state(run_dir, state)


# --------------------------------------------------------------------------- #
# the derived "used by"
# --------------------------------------------------------------------------- #
def test_used_by_names_the_comparison_that_read_the_snapshot(tmp_path):
    """The column is computed from `comparisons/`, not stored on the snapshot — so this asserts a
    real id appears, never merely that a column rendered."""
    exp = _cohort(tmp_path)
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["animal"], "aggregate_to": []}
    )
    snapshot_id = comp.record()["source_snapshot"]

    assert exp.comparisons_using("an_01") == {snapshot_id: [comp.id]}
    rows = snapshot_rows(exp, _run(exp))
    assert any(comp.id in row[-1] for row in rows)


def test_a_snapshot_nothing_used_says_so_rather_than_naming_something(tmp_path):
    exp = _cohort(tmp_path)
    _run(exp).pin(reason="finish")
    rows = snapshot_rows(exp, _run(exp))
    assert len(rows) == 1 and rows[0][-1] == "—"


def test_a_comparison_on_another_run_is_not_attributed_here(tmp_path):
    """The negative case: without the `source_run_id` filter every comparison in the experiment
    would be credited to every run."""
    exp = _cohort(tmp_path)
    _add_run(exp, "an_02")  # a second run in the same experiment
    comp = run_comparison(
        exp, {"source_run_id": "an_02", "group_by": ["animal"], "aggregate_to": []}
    )
    assert comp.id not in str(exp.comparisons_using("an_01"))
    assert comp.id in str(exp.comparisons_using("an_02"))


def test_a_record_written_before_snapshots_is_skipped_not_guessed(tmp_path):
    """Comparisons already on disk carry no `source_snapshot`; they must not be attributed to an
    arbitrary snapshot."""
    from detail.storage import comparisons as comparisons_store

    exp = _cohort(tmp_path)
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["animal"], "aggregate_to": []}
    )
    record = comp.record()
    del record["source_snapshot"]
    comparisons_store.write_comparison_yaml(comp.dir, record)

    assert exp.comparisons_using("an_01") == {}


# --------------------------------------------------------------------------- #
# the surfaces
# --------------------------------------------------------------------------- #
def test_the_row_does_not_follow_the_run_it_came_from(tmp_path):
    """The whole point of a pinned copy. The instance count is read from the frozen artifact, so a
    reprojection underneath must not move it."""
    exp = _cohort(tmp_path)
    run = _run(exp)
    run.pin(reason="finish")
    before = snapshot_rows(exp, run)[0]
    assert before[4] == "6", "instances, after the module's own columns"

    run_dir = analysis_runs.analysis_dir(exp.dir, "an_01")
    artifact = bo.read_behavior_output(run_dir)
    bo.write_behavior_output(
        run_dir,
        instances=list(bo.bouts(artifact.instances))[:1],
        coverage=list(artifact.coverage),
        manifest=artifact.manifest,
    )
    assert snapshot_rows(exp, run)[0] == before


def test_the_row_names_the_vocabulary_it_was_taken_at(tmp_path):
    """What a person is actually trying to recall — "which of these was the one with the three
    postures?" — and the reason the row carries it rather than a detail view."""
    exp = _cohort(tmp_path)
    run = _run(exp)
    _name(exp, {"c01": "rearing"})
    run.pin(reason="finish")

    taken, why, labels, vocabulary, instances, used = snapshot_rows(exp, run)[0]
    assert why == "finish"
    assert labels == "1" and vocabulary == "rearing"
    assert instances == "6"
    columns = snapshot_columns(run, run.snapshots())
    assert columns == [
        "Taken",
        "Why",
        "Labels",
        "Vocabulary",
        "Instances",
        "Used by",
    ]


def test_several_clusters_sharing_a_name_are_listed_once(tmp_path):
    """§22 makes many-to-one naming the normal case, so a vocabulary that repeated "rearing" four
    times would be noise rather than information."""
    exp = _cohort(tmp_path)
    run = _run(exp)
    _name(exp, {"c01": "rearing", "c02": "rearing", "c03": "grooming"})
    run.pin(reason="finish")
    assert snapshot_rows(exp, run)[0][3] == "rearing, grooming"


def test_a_long_vocabulary_is_cut_with_a_count(tmp_path):
    """A row is a row; twenty names would push everything else off it."""
    exp = _cohort(tmp_path)
    run = _run(exp)
    _name(exp, {f"c{i:02d}": f"behavior {i}" for i in range(7)})
    run.pin(reason="finish")
    vocabulary = snapshot_rows(exp, run)[0][3]
    assert vocabulary.endswith("+3 more") and vocabulary.count(",") == 3


def test_an_unnamed_interpretation_says_unnamed_rather_than_nothing(tmp_path):
    exp = _cohort(tmp_path)
    run = _run(exp)
    run.pin(reason="export")
    assert snapshot_rows(exp, run)[0][3] == "unnamed"


def test_a_run_that_has_pinned_nothing_shows_no_table(tmp_path):
    """An empty table would suggest something is missing; nothing has left yet."""
    exp = _cohort(tmp_path)
    assert snapshot_rows(exp, _run(exp)) == []


def test_a_comparison_pinned_snapshot_records_its_interpretation(tmp_path):
    """Pinning by directory alone left `state: {}`, so Labels / Vocabulary were blank on
    exactly the snapshots most worth reading later — the ones a comparison used. It pins through
    the run handle now."""
    exp = _cohort(tmp_path)
    _name(exp, {"c01": "rearing"})
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["animal"], "aggregate_to": []}
    )
    row = snapshot_rows(exp, _run(exp))[0]
    assert row[1] == "comparison"
    assert row[2] == "1", "the labels behind it"
    assert row[3] == "rearing", "the vocabulary it measured"
    assert comp.id in row[5]


# --------------------------------------------------------------------------- #
# snapshot mode on the run page (§21)
# --------------------------------------------------------------------------- #
def _page(exp, run_id="an_01"):
    """The run detail page, held by the caller.

    Held in a local by every caller: a page left as a temporary is collected and takes its widgets
    with it, which surfaces as "wrapped C/C++ object ... has been deleted" rather than as a clear
    failure.
    """
    from detail.gui.pages import AnalysisRunDetailPage

    return AnalysisRunDetailPage(exp, run_id)


def _live(page, cls):
    """Widgets the page is showing **now**, by class.

    Walks `_body_widgets` rather than `findChildren`: `_build()` retires the old body with
    `deleteLater()`, so until the event loop runs `findChildren` still returns the previous
    render's widgets — and a test written against it passes while asserting about a screen that is
    gone.
    """
    out = []
    for widget in page._body_widgets:
        if isinstance(widget, cls):
            out.append(widget)
        out.extend(widget.findChildren(cls))
    return out


def _snapshot_table(page):
    from detail.gui.qt import QTableView

    for table in _live(page, QTableView):
        model = table.model()
        headers = [
            model.horizontalHeaderItem(c).text() for c in range(model.columnCount())
        ]
        if "Taken" in headers and "Used by" in headers:
            return table
    return None


def _banner_text(page) -> str:
    from detail.gui.qt import QLabel

    return " ".join(
        label.text() for label in _live(page, QLabel) if "Viewing snap" in label.text()
    )


def _open_first_snapshot(page):
    table = _snapshot_table(page)
    page._open_snapshot_row(table.model().index(0, 0))
    return page


def test_opening_a_snapshot_puts_the_page_in_that_snapshots_mode(qapp, tmp_path):
    """The whole flow in one assertion: the page does not navigate anywhere, it becomes the
    snapshot."""
    exp = _cohort(tmp_path)
    run = _run(exp)
    _name(exp, {"c01": "rearing"})
    snapshot_id = run.pin(reason="finish")

    page = _page(exp)
    assert page._viewing is None
    _open_first_snapshot(page)

    assert page._viewing == snapshot_id
    banner = _banner_text(page)
    assert snapshot_id in banner and "Read-only" in banner
    assert "rearing" in banner, "the banner names the interpretation, not just the id"


def test_the_mode_survives_a_rebuild(qapp, tmp_path):
    """`_build()` re-renders a shown page. If the mode were local to the click handler an
    incidental refresh would drop the page silently back to the live run — same layout, different
    numbers, no event — which is this feature's own failure mode arriving through a side door.
    """
    exp = _cohort(tmp_path)
    snapshot_id = _run(exp).pin(reason="export")
    page = _page(exp)
    _open_first_snapshot(page)

    page._build()

    assert page._viewing == snapshot_id
    assert snapshot_id in _banner_text(page)


def test_going_back_returns_to_the_live_run(qapp, tmp_path):
    exp = _cohort(tmp_path)
    _run(exp).pin(reason="export")
    page = _page(exp)
    _open_first_snapshot(page)

    page._back_to_live()

    assert page._viewing is None
    assert _banner_text(page) == ""


def test_the_row_being_read_is_marked_in_the_list(qapp, tmp_path):
    """ "Which one am I on" is answered where the list is, as well as in the banner."""
    exp = _cohort(tmp_path)
    run = _run(exp)
    run.pin(reason="export")
    # A second pin needs a *different* artifact: pin is minted by difference, not by action, so
    # naming alone (which does not change the instances here, there being no fit to reproject
    # through) would return the first snapshot again.
    _name(exp, {"c01": "rearing"})
    artifact = bo.read_behavior_output(analysis_runs.analysis_dir(exp.dir, "an_01"))
    bo.write_behavior_output(
        analysis_runs.analysis_dir(exp.dir, "an_01"),
        instances=list(bo.bouts(artifact.instances))[:2],
        coverage=list(artifact.coverage),
        manifest=artifact.manifest,
    )
    run.pin(reason="finish")

    page = _page(exp)
    table = _snapshot_table(page)
    assert table.model().rowCount() == 2
    page._open_snapshot_row(table.model().index(1, 0))

    model = _snapshot_table(page).model()
    assert model.item(1, 0).font().bold()
    assert not model.item(0, 0).font().bold()


def test_the_live_runs_actions_are_greyed_while_a_snapshot_is_read(qapp, tmp_path):
    """Greyed, not removed: they are capabilities that do not apply in this mode. Deleting the run
    from under the snapshot you are reading is the one that matters."""
    from detail.gui.qt import QPushButton

    exp = _cohort(tmp_path)
    _run(exp).pin(reason="export")
    page = _page(exp)
    live = {b.text(): b.isEnabled() for b in _live(page, QPushButton)}
    assert live.get("Delete analysis") is True

    _open_first_snapshot(page)
    frozen = {b.text(): b for b in _live(page, QPushButton)}
    assert "Delete analysis" in frozen, "greyed out, not hidden"
    assert not frozen["Delete analysis"].isEnabled()
    assert "Back to live run" in frozen and frozen["Back to live run"].isEnabled()


def test_the_live_measures_panel_is_not_shown_under_a_snapshot_banner(qapp, tmp_path):
    """Hidden rather than greyed, because its numbers describe the live run — live numbers under a
    snapshot banner are the exact confusion this feature exists to prevent."""
    from detail.gui.qt import QLabel

    exp = _cohort(tmp_path)
    _run(exp).pin(reason="export")
    page = _page(exp)

    from detail.gui.qt import QTabWidget

    def shown():
        """Measures as a page heading or as a workspace tab — wherever this run puts it."""
        tabs = {
            tabs.tabText(i)
            for tabs in _live(page, QTabWidget)
            for i in range(tabs.count())
        }
        return {label.text() for label in _live(page, QLabel)} | tabs

    assert {"Measures", "MEASURES"} & shown()

    _open_first_snapshot(page)
    assert not {"Measures", "MEASURES"} & shown()


def test_a_snapshot_with_no_recorded_settings_says_so_instead_of_opening(
    qapp, tmp_path, monkeypatch
):
    """Real runs on disk have these — pinned before a comparison pinned through the run handle."""
    from detail.analysis import provenance
    from detail.gui import pages as pages_module
    from detail.storage import snapshots as snapshots_store

    exp = _cohort(tmp_path)
    snapshots_store.pin(
        analysis_runs.analysis_dir(exp.dir, "an_01"),
        reason="comparison",
        code_sha=provenance.code_version(),
    )
    shown = []
    monkeypatch.setattr(
        pages_module.QMessageBox,
        "information",
        lambda parent, title, text, *a, **k: shown.append(text),
    )

    page = _page(exp)
    _open_first_snapshot(page)

    assert page._viewing is None, "it did not enter a mode it cannot render"
    assert shown and "recorded no settings" in shown[0]


def test_the_banner_is_centred_and_stands_off_the_page(qapp, tmp_path):
    """It has one job: be impossible to read past. Centred, amber, on a band that is plainly not
    the page — the shade derived from the current Window colour so it separates in a light theme
    and a dark one, since a fixed grey vanishes into one of them.

    Amber rather than decoration: "you are not looking at the live run" is a status, which is the
    only thing colour is spent on here (`gui/status_color.py`). Through the palette, never a
    stylesheet — the GUI has no QSS anywhere.
    """
    from detail.gui.qt import QFrame, QLabel, QPalette, Qt

    from detail.gui.status_color import WARN

    exp = _cohort(tmp_path)
    _run(exp).pin(reason="export")
    page = _page(exp)
    _open_first_snapshot(page)

    label = next(l for l in _live(page, QLabel) if "Viewing snap" in l.text())
    assert label.alignment() & Qt.AlignmentFlag.AlignHCenter
    assert label.palette().color(QPalette.ColorRole.WindowText) == WARN
    assert label.font().bold()

    banner = label.parentWidget()
    assert isinstance(banner, QFrame) and banner.autoFillBackground()
    page_shade = page.palette().color(QPalette.ColorRole.Window).lightness()
    banner_shade = banner.palette().color(QPalette.ColorRole.Window).lightness()
    assert (
        abs(page_shade - banner_shade) > 10
    ), "the band is not distinguishable from the page"

    assert not banner.styleSheet(), "no QSS anywhere in the GUI (docs/gui_design.md)"


def test_the_snapshot_row_is_reachable_by_double_click(qapp, tmp_path):
    """Through the real entry point, not the handler behind it.

    Every other test here calls `page._open_snapshot_row(index)`. The wiring it skips is
    conditional — `openable` decides the chevron, the hint and whether `doubleClicked` is connected
    at all — so forcing `openable = False`, which makes the feature unreachable in the app, left 58
    tests green.
    """
    exp = _cohort(tmp_path)
    snapshot_id = _run(exp).pin(reason="export")
    page = _page(exp)
    table = _snapshot_table(page)

    headers = [
        table.model().horizontalHeaderItem(c).text()
        for c in range(table.model().columnCount())
    ]
    assert headers[-1] == "", "the trailing chevron column marks the row as navigable"

    table.doubleClicked.emit(table.model().index(0, 0))

    assert page._viewing == snapshot_id, "the signal reaches the handler"
    assert snapshot_id in _banner_text(page)


def test_a_module_that_cannot_render_a_snapshot_gets_no_cue(
    qapp, tmp_path, monkeypatch
):
    """The other half of the same rule: no chevron where there is nothing to open."""
    from detail.api.temporal_clustering import TemporalClusteringRun

    exp = _cohort(tmp_path)
    _run(exp).pin(reason="export")
    monkeypatch.setattr(TemporalClusteringRun, "snapshot_view_class", None)

    page = _page(exp)
    table = _snapshot_table(page)
    headers = [
        table.model().horizontalHeaderItem(c).text()
        for c in range(table.model().columnCount())
    ]
    assert headers[-1] != "", "no trailing chevron column"
    from detail.gui.qt import QLabel

    # The snapshot section's own phrase, not the workspace's "Double-click a row to name it".
    blurb = " ".join(label.text() for label in _live(page, QLabel))
    assert "Double-click one to see what it was" not in blurb
