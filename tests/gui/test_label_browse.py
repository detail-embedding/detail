"""BROWSE on a clustering run: the labelled dataset, filtered and watched (§21/§22).

The pure half — which bouts a filter keeps, how the label filter nests, where the ethogram's
strips come from — is asserted without Qt. The widget half asserts what a user does: untick a
label, click a band, select rows, export. The fixture artifact is written by hand, because what
is under test is the reading of label paths and filters, and a real fit would make the label mix
an accident of the fixture; the tab's life inside a real workspace is
`tests/gui/test_cluster_workspace.py`.
"""

import types

import pytest

from detail.gui.label_browse import (
    NO_VALUE,
    ROW_CAP,
    LabelBrowseTab,
    filter_bouts,
    label_table,
    labelled_bouts,
    recording_passes,
)
from detail.gui.qt import Qt
from detail.gui.sizing import lines
from detail.storage import behavior_output as bo

from tests.gui.conftest import drain_busy_calls

#: ``(source label, path)`` cycled through by `labelled_output`; the last is the remainder.
_SHAPES = [
    ("101", "rearing"),
    ("102", "rearing/supported"),
    ("103", "grooming"),
    ("unlabelled", ""),
]


def _bout(i: int, path: str, recording_id: str = "recording_001", source="n"):
    return bo.BehaviorInstance(
        recording_id=recording_id,
        subject=0,
        t_ns=i * 1_000_000_000,
        end_t_ns=i * 1_000_000_000 + 400_000_000,
        start_frame=i,
        end_frame=i,
        source_label=source,
        label=path.rsplit("/", 1)[-1] if path else "",
        label_path=path,
    )


def labelled_output(recording_id: str, n: int, recordings=None) -> bo.BehaviorOutput:
    """An artifact of ``n`` labelled bouts (plus a remainder bout per three), as a run reads it.

    Spread across ``recordings`` when given, so the recording and metadata filters have
    something to choose between.
    """
    recordings = recordings or [recording_id]
    rows = []
    for i in range(n + n // 3):
        source, path = _SHAPES[i % 4]
        rows.append(_bout(i, path, recordings[i % len(recordings)], source))
    labelled = [r for r in rows if r.label][:n]
    remainder = [r for r in rows if not r.label]
    coverage = [
        bo.CoverageRow(r, 0, 0, (n + n // 3 + 1) * 1_000_000_000, 10**12, 1000)
        for r in recordings
    ]
    return bo.BehaviorOutput(
        manifest={}, instances=labelled + remainder, coverage=coverage
    )


# --------------------------------------------------------------------------- #
# the dataset, without Qt
# --------------------------------------------------------------------------- #
def test_the_dataset_is_labelled_bouts_only():
    rows = labelled_bouts(labelled_output("r1", 9).instances)
    assert len(rows) == 9
    assert all(row.label for row in rows), "the unlabelled remainder is not the dataset"


def test_a_label_filter_is_exact_so_a_parent_and_its_sub_label_part():
    """Ticking "rearing" and not "supported" shows rearing's other bouts only."""
    rows = [_bout(1, "rearing"), _bout(2, "rearing/supported"), _bout(3, "grooming")]
    kept = filter_bouts(rows, paths={"rearing"})
    assert [r.label_path for r in kept] == ["rearing"]
    assert filter_bouts(rows) == rows, "no filter keeps everything"


def test_the_metadata_filter_files_a_missing_value_under_its_own_name():
    metadata = {"r1": {"treatment": "vehicle"}, "r2": {}}
    common = dict(meta_column="treatment", metadata=metadata)
    assert recording_passes("r1", meta_values={"vehicle"}, **common)
    assert not recording_passes("r2", meta_values={"vehicle"}, **common)
    assert recording_passes("r2", meta_values={NO_VALUE}, **common)
    assert recording_passes("r3", **common), "a column alone filters nothing"
    assert not recording_passes("r1", recordings={"r2"})


def test_the_label_filter_nests_like_the_labels_panel():
    """A parent's row counts its whole group; beneath it each sub-label, and "other" for its
    own bouts — the rows INTERPRET's labels panel draws, so "other" means one thing."""
    rows = [_bout(i, "rearing/supported") for i in range(3)] + [
        _bout(10, "rearing"),
        _bout(11, "grooming"),
    ]
    table = label_table(rows, {"recording_001": 100_000_000})
    shape = [
        (text, depth, bouts, path) for text, depth, _root, bouts, _t, path in table
    ]
    assert shape == [
        ("rearing", 0, 4, "rearing"),
        ("supported", 1, 3, "rearing/supported"),
        ("other", 1, 1, "rearing"),
        ("grooming", 0, 1, "grooming"),
    ]


# --------------------------------------------------------------------------- #
# the tab
# --------------------------------------------------------------------------- #
_RECORDINGS = ["recording_001", "recording_002"]


def _run(output, *, editable=True):
    """A run handle with just what BROWSE reads. A test points ``_audit_dir`` at its own audit
    log, and ``audit_decisions`` reads whatever it points at, as the real handle's does.
    """
    from detail.storage import behavior_audit as audit

    run = types.SimpleNamespace(
        id="an_01",
        editable=editable,
        artifact=lambda: output,
        data_sources=lambda: {
            "recording_ids": list(_RECORDINGS),
            "embed_run_ids": {rid: "emb_01" for rid in _RECORDINGS},
        },
        _audit_dir=lambda: "/nonexistent-audit-dir",
    )
    run.audit_decisions = lambda: audit.latest_decisions(run._audit_dir())
    return run


@pytest.fixture
def tab(qapp, tmp_path):
    from detail.gui.cluster_workspace import TEMPORAL_TERMS

    experiment = types.SimpleNamespace(
        dir=tmp_path,
        recording_names=lambda: {"recording_001": "Tank A", "recording_002": "Tank B"},
        recording_metadata={
            "recording_001": {"treatment": "vehicle"},
            "recording_002": {"treatment": "drug"},
        },
        metadata_columns=["treatment"],
    )
    output = labelled_output("recording_001", 12, recordings=_RECORDINGS)
    widget = LabelBrowseTab(_run(output), experiment, TEMPORAL_TERMS)
    drain_busy_calls(qapp)
    assert widget._model.rowCount() == 12
    return widget


def _paths_shown(tab) -> set[str]:
    return {row.label_path for row in tab._rows}


def test_unticking_a_label_takes_it_out_of_the_list_the_ethogram_and_the_count(tab):
    grooming = next(
        item
        for item in tab._label_items()
        if item.data(0, Qt.ItemDataRole.UserRole) == "grooming"
    )
    grooming.setCheckState(0, Qt.CheckState.Unchecked)
    assert "grooming" not in _paths_shown(tab)
    assert tab._model.rowCount() == 8
    assert all(
        row.label_path != "grooming" for rows in tab._band_rows for row in rows
    ), "the ethogram still draws an unticked label"
    assert "8 of 12" in tab._summary.text() and "filters:" in tab._summary.text()
    assert tab._clear.isEnabled()

    tab._clear_filters()
    assert tab._model.rowCount() == 12 and not tab._clear.isEnabled()


def test_a_metadata_group_narrows_the_list_and_the_strips(tab):
    tab._column.setCurrentIndex(tab._column.findData("treatment"))
    drug = next(
        tab._values.item(i)
        for i in range(tab._values.count())
        if tab._values.item(i).data(Qt.ItemDataRole.UserRole) == "drug"
    )
    drug.setCheckState(Qt.CheckState.Unchecked)
    assert {row.recording_id for row in tab._rows} == {"recording_001"}
    assert len(tab._band_rows) == 1, "a recording outside the group still has a strip"
    assert "treatment = vehicle" in tab._summary.text()


def test_clicking_a_band_selects_its_row_and_selecting_a_row_outlines_its_band(tab):
    target = tab._band_rows[1][2]
    tab._on_band(1, 2)
    current = tab._table.currentIndex().row()
    assert tab._row_at(current).instance_id == target.instance_id
    assert tab._ethogram._highlight == (1, 2)

    tab._table.selectRow(0)
    first = tab._row_at(0)
    assert tab._ethogram._highlight == tab._band_of[first.instance_id]


def test_a_band_past_the_row_cap_still_plays(tab, monkeypatch):
    """The ethogram draws every filtered bout and the list only the first `ROW_CAP`; a band the
    list has no row for plays directly rather than doing nothing."""
    played = []
    monkeypatch.setattr(tab, "_play", played.append)
    tab._rows = tab._rows[:1]
    target = tab._band_rows[1][-1]
    tab._model.removeRows(1, tab._model.rowCount() - 1)
    tab._on_band(1, len(tab._band_rows[1]) - 1)
    assert played == [target]
    assert ROW_CAP >= 1000, "the cap is a build cost, not a preference"


def test_the_export_scope_is_the_selection_and_export_all_is_every_filtered_bout(
    tab, monkeypatch
):
    from detail.gui.qt import QItemSelectionModel

    assert tab.export_scope() is None
    assert not tab._export_selected.isEnabled()
    model = tab._table.selectionModel()
    for row in (0, 3):
        model.select(
            tab._model.index(row, 0),
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows,
        )
    picked = {tab._row_at(0).instance_id, tab._row_at(3).instance_id}
    assert set(tab.export_scope().instance_ids) == picked
    assert tab._export_selected.text() == "Export 2 selected"

    opened = []
    monkeypatch.setattr(
        "detail.gui.export.open_export",
        lambda parent, subject, scope, **kw: opened.append((subject, scope)),
    )
    tab._export(selected=False)
    subject, scope = opened[0]
    assert (subject.kind, subject.id) == ("analysis_run", "an_01")
    assert set(scope.instance_ids) == {row.instance_id for row in tab._shown}


def test_the_audit_column_says_what_the_reviewer_answered(qapp, tmp_path):
    """A judged instance shows the level the reviewer picked, in the words they picked it by."""
    from detail.gui.cluster_workspace import TEMPORAL_TERMS
    from detail.storage import behavior_audit as audit

    output = labelled_output("recording_001", 3)
    supported = next(r for r in output.instances if r.label_path == "rearing/supported")
    audit.append_audit(
        tmp_path, supported.instance_id, human_label=False, answer="rearing"
    )
    run = _run(output)
    run._audit_dir = lambda: tmp_path
    experiment = types.SimpleNamespace(
        dir=tmp_path,
        recording_names=lambda: {},
        recording_metadata={},
        metadata_columns=[],
    )
    widget = LabelBrowseTab(run, experiment, TEMPORAL_TERMS)
    drain_busy_calls(qapp)
    audit_column = widget._model.columnCount() - 1
    said = {
        widget._row_at(r).label_path: widget._model.item(r, audit_column).text()
        for r in range(widget._model.rowCount())
    }
    assert said["rearing/supported"] == "rearing, not supported"
    assert said["rearing"] == "—"
    assert widget._no_metadata.isVisibleTo(widget)


def test_nothing_labelled_says_where_labels_come_from(qapp, tmp_path):
    from detail.gui.cluster_workspace import TEMPORAL_TERMS

    output = labelled_output("recording_001", 0)
    experiment = types.SimpleNamespace(
        dir=tmp_path,
        recording_names=lambda: {},
        recording_metadata={},
        metadata_columns=[],
    )
    widget = LabelBrowseTab(_run(output), experiment, TEMPORAL_TERMS)
    drain_busy_calls(qapp)
    assert widget._model.rowCount() == 0
    # One message in place of the whole screen — not empty filters, strips, list and player.
    assert widget._pages.currentIndex() == 0
    assert "INTERPRET" in widget._empty_body.text()


def test_a_snapshot_reads_and_filters_but_does_not_export(qapp, tmp_path):
    from detail.gui.cluster_workspace import TEMPORAL_TERMS

    experiment = types.SimpleNamespace(
        dir=tmp_path,
        recording_names=lambda: {},
        recording_metadata={},
        metadata_columns=[],
    )
    widget = LabelBrowseTab(
        _run(labelled_output("recording_001", 4), editable=False),
        experiment,
        TEMPORAL_TERMS,
    )
    drain_busy_calls(qapp)
    assert widget._model.rowCount() == 4
    assert not widget._export_all.isEnabled()
    assert "snapshot" in widget._export_all.toolTip()


def test_the_tab_opens_on_the_first_instance_paused_and_unselected(tab):
    """The player shows what the list is of, without anyone asking — still, and without
    selecting the row, because a selected row is an export scope nobody picked."""
    assert tab._playing == tab._rows[0].instance_id
    assert tab._player._start_on_first_frame is False, "opened playing"
    assert tab.export_scope() is None
    assert not tab._table.selectionModel().hasSelection()


def test_filters_that_leave_nothing_replace_the_results_with_one_sentence(tab):
    for item in tab._items():
        item.setCheckState(0, Qt.CheckState.Unchecked)
    assert tab._results.currentIndex() == 1, "an empty picture, list and player"
    tab._clear_filters()
    assert tab._results.currentIndex() == 0
    assert tab._playing in {
        row.instance_id for row in tab._shown
    }, "the player kept a clip the filters had taken away"


def test_the_ethogram_handle_shows_only_when_some_strips_are_hidden(qapp, tmp_path):
    """A drag bar that moves nothing reads as broken: with every strip on screen there is
    nothing to drag toward."""
    from detail.gui.cluster_workspace import TEMPORAL_TERMS

    def build(n_recordings):
        recordings = [f"recording_{i:03d}" for i in range(n_recordings)]
        experiment = types.SimpleNamespace(
            dir=tmp_path,
            recording_names=lambda: {},
            recording_metadata={},
            metadata_columns=[],
        )
        widget = LabelBrowseTab(
            _run(labelled_output(recordings[0], 30, recordings=recordings)),
            experiment,
            TEMPORAL_TERMS,
        )
        drain_busy_calls(qapp)
        widget.resize(1200, 800)
        widget.show()
        qapp.processEvents()
        return widget

    few, many = build(2), build(7)
    try:
        assert few._ethogram_band._handle.isHidden()
        assert not many._ethogram_band._handle.isHidden()
        assert many._ethogram_scroll.height() < many._ethogram.height()
    finally:
        few.hide()
        many.hide()


def test_the_filter_column_never_ends_below_the_results(qapp, tab):
    """The filter lists share the results' height and never push the column past it: three
    ~192px lists made the rail the taller one, with dead space under the results. Measured
    against the export buttons rather than the results pane: a splitter gives both panes one
    height whatever is in them, so comparing panes passes on the very layout this rules out.

    It used to be pinned *level* with the buttons as well; the lists now stop at their last row
    (`test_a_rail_list_is_no_taller_than_its_rows`), so the rail may end above them."""
    # Filtered to one label so the results are *shorter* than three lists at their own
    # height — at the fixture's full twelve rows the two happen to match, and the check passes
    # whether the lists share height or not.
    for item in tab._label_items():
        if item.data(0, Qt.ItemDataRole.UserRole) != "grooming":
            item.setCheckState(0, Qt.CheckState.Unchecked)
    tab.resize(1500, 900)
    tab.show()
    qapp.processEvents()
    try:
        # The lowest thing drawn in the column, not the column: the column's own widget fills
        # its pane whatever is in it, so measuring it passes on a rail that stops halfway.
        drawn = [
            w
            for w in (
                tab._labels,
                tab._recordings,
                tab._column,
                tab._values,
                tab._no_metadata,
            )
            if w.isVisibleTo(tab)
        ]
        rail_bottom = max(w.mapTo(tab, w.rect().bottomLeft()).y() for w in drawn)
        button = tab._export_all
        button_bottom = button.mapTo(tab, button.rect().bottomLeft()).y()
        assert rail_bottom <= button_bottom + lines(1), (rail_bottom, button_bottom)
    finally:
        tab.hide()


# --------------------------------------------------------------------------- #
# laptop sizes (the 1440x847 render audit)
# --------------------------------------------------------------------------- #
def test_a_rail_list_is_no_taller_than_its_rows(qapp, tab):
    """Measured at 1440x847: Labels was a 290px box for four rows and Recordings 210px for two,
    because each list took a share of the results' height whatever it held. A list now stops
    at its last row; the rail's surplus collects below the lists."""
    tab.resize(1500, 900)
    tab.show()
    qapp.processEvents()
    try:
        for widget in (tab._labels, tab._recordings):
            rows = LabelBrowseTab.rows_px(widget)
            assert rows < 200, "the fixture must be short enough for the check to bite"
            assert widget.height() <= max(rows, widget.minimumHeight()) + 1, (
                widget.height(),
                rows,
            )
        tab._search.setText("Tank A")
        qapp.processEvents()
        assert tab._recordings.height() <= max(
            LabelBrowseTab.rows_px(tab._recordings), tab._recordings.minimumHeight()
        )
    finally:
        tab.hide()


def test_the_ethogram_keeps_every_strip_when_the_cluster_lane_arrives(qapp, tab):
    """INTERPRET's lane grows every row of the strip, after the band was sized to the rows
    without it — at 1440x847 the second of two recordings sat half under a scrollbar. The band
    follows the strip's height, so two strips are two strips, whole."""
    tab.resize(1500, 900)
    tab.show()
    qapp.processEvents()
    try:
        before = tab._ethogram.height()
        runs = [
            {"recording_id": rid, "start_ns": 0, "end_ns": 1_000_000_000}
            for rid in _RECORDINGS
        ]
        view = types.SimpleNamespace(labelled_runs=lambda *_a, **_k: list(runs))
        tab.set_cluster_lane((view, 7, "climb", plot_theme_colour()))
        from tests.gui.test_query_workspace import _drain

        assert _drain(qapp, lambda: tab._lane is not None)
        qapp.processEvents()
        assert tab._ethogram.height() > before, "the lane did not grow the rows"
        assert tab._ethogram_scroll.height() >= tab._ethogram.height()
        assert tab._ethogram_scroll.verticalScrollBar().maximum() == 0
    finally:
        tab.hide()


def plot_theme_colour():
    from detail.gui.qt import QColor

    return QColor("#1f77b4")


def test_the_lists_last_column_is_whole_beside_the_player(qapp, tab):
    """A fixed 520/460 split cut the audit column to "aud" at 1440x847, behind a horizontal
    scrollbar. The list takes its columns' width first, the player keeps a floor, and the name
    column gives up what is still short — eliding, its whole text on hover."""
    from detail.gui.qt import QSplitter

    tab.resize(1000, 800)
    tab.show()
    qapp.processEvents()
    try:
        table = tab._table
        header = table.horizontalHeader()
        last = header.count() - 1
        assert table.horizontalScrollBar().maximum() == 0, "the list scrolls sideways"
        right = header.sectionViewportPosition(last) + header.sectionSize(last)
        assert right <= table.viewport().width()
        assert header.sectionSize(last) >= table.sizeHintForColumn(last) - 1
        name = tab._model.item(0, 0)
        assert name.toolTip() == name.text()
        # A divider the user dragged is theirs: a resize does not refit it to the columns
        # (the splitter only shares the new width out in proportion).
        split = tab._list_split
        assert isinstance(split, QSplitter)
        fitted = split.sizes()[0]
        assert (
            fitted > 400
        ), "the fixture's columns must be wider than the dragged split"
        split.splitterMoved.emit(250, 1)
        split.setSizes([250, sum(split.sizes()) - 250])
        tab.resize(1100, 800)
        qapp.processEvents()
        assert split.sizes()[0] < 300
    finally:
        tab.hide()


def test_the_cluster_lane_is_read_only_once_browse_is_shown_and_off_the_ui_thread(
    qapp, tab
):
    """Every INTERPRET node click handed BROWSE a lane, and BROWSE walked every leaf run through
    ``labelled_runs`` on the UI thread at once — while it was a hidden tab nobody was looking at.
    The request is remembered while hidden, read when BROWSE shows, and read on a worker; only
    the last request of several lands."""
    import threading

    from tests.gui.test_query_workspace import _drain

    calls = []

    def labelled_runs(names, **_kw):
        calls.append((names, threading.current_thread() is threading.main_thread()))
        return [
            {"recording_id": rid, "start_ns": 0, "end_ns": 1_000_000_000}
            for rid in _RECORDINGS
        ]

    view = types.SimpleNamespace(labelled_runs=labelled_runs)
    tab.hide()
    qapp.processEvents()
    tab.set_cluster_lane((view, 7, "climb", plot_theme_colour()))
    tab.set_cluster_lane((view, 8, "rear", plot_theme_colour()))
    qapp.processEvents()
    assert calls == [], "read while BROWSE was hidden"
    tab.resize(1500, 900)
    tab.show()
    try:
        assert _drain(qapp, lambda: tab._lane is not None)
        assert [c[0] for c in calls] == [{"8": "rear"}], "only the last request is read"
        assert calls[0][1] is False, "read on the UI thread"
        assert tab._lane[0] == "rear"
        assert sum(len(m) for m in tab._ethogram._lane) == len(_RECORDINGS)
        tab.set_cluster_lane(None)
        assert tab._lane is None and tab._ethogram._lane is None
    finally:
        tab.hide()


def test_the_key_and_the_bands_tell_the_labels_apart_without_their_hue(qapp, tab):
    """A colour-blind reader must be able to tell "grooming" from "rearing" on BROWSE: the
    rail's swatch, the list's swatch and the ethogram's bands each carry the label's pattern as
    well as its colour, and the list keys a row with exactly the rail's swatch for its label
    (`tests/gui/test_identity_marks.py` is the class; this is the page)."""
    import numpy as np
    from detail.gui.qt import QPixmap

    from tests.gui import greyscale

    rail = {}
    for i in range(tab._labels.topLevelItemCount()):
        item = tab._labels.topLevelItem(i)
        swatch = item.data(0, Qt.ItemDataRole.DecorationRole)
        assert isinstance(swatch, QPixmap), f"{item.text(0)}'s key is {swatch!r}"
        rail[item.text(0)] = greyscale.luminance(swatch)
    assert set(rail) == {"grooming", "rearing"}
    assert greyscale.apart(rail["grooming"], rail["rearing"])

    for view_row in range(tab._model.rowCount()):
        root = tab._row_at(view_row).label_path.split("/")[0]
        listed = tab._model.item(view_row, 0).data(Qt.ItemDataRole.DecorationRole)
        assert np.array_equal(greyscale.luminance(listed), rail[root])

    tab.resize(1440, 847)
    tab.show()
    qapp.processEvents()
    try:
        strip = tab._ethogram
        strip.set_view(0.0, 0.25)  # room for a pattern: a whole-span bout is ~20px
        qapp.processEvents()
        image = strip.grab().toImage()
        bands = {}
        for row, owners in enumerate(tab._band_rows):
            for index, bout in enumerate(owners):
                root = bout.label_path.split("/")[0]
                start, end, _slot = strip._rows[row][1][index]
                x0, x1 = strip._x(start, row) + 2, strip._x(end, row) - 2
                if root not in bands and x1 - x0 >= 32 and x0 >= strip.track()[0]:
                    bands[root] = greyscale.luminance(
                        image.copy(x0, strip._bar_top(row), x1 - x0, strip.BAR_PX)
                    )
    finally:
        tab.hide()
    assert sorted(bands) == ["grooming", "rearing"], "no wide band of each was drawn"
    assert greyscale.apart(bands["grooming"], bands["rearing"], shift=8)
