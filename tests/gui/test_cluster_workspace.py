"""The interpretation workspace a clustering run is labelled through (§21/§22).

INTERPRET is the label field and the filters over the tree, and every clip of the selected
cluster playing under it. What these assert is what is on screen and what reaches the run: which
node is selected, what the montage plays and in what order, how a filter shades the tree and greys
the clips, that a label lands in the run's state and its output, and that a snapshot or a finished
run has nothing live that would change it. The run is
real: the session's kinematics fit (``fitted``), which writes its node cache in its ``caching``
phase.

One recorded class is guarded here: **a widget emptied for a pending load while the page's own
copy of what it shows is not.** A bout method ranks a cluster's clips off the UI thread, and the
montage was emptied for the wait while ``_clips`` / ``_order`` still held the previous cluster's.
Every control that acts on the montage reads that copy, so a filter set in the wait handed 21
flags to a montage of 0 clips (``ValueError`` in a slot) and the order and context switches drew
the previous cluster's tiles back under "ranking…". The background result was never the problem —
its request token already dropped a stale one; the stale state was the page's own. The page and
its montage are emptied together (``node_workspace._clear_clips``; the cluster window does the
same). Guarded by ``test_a_filter_set_while_a_cluster_ranks_waits_for_its_clips``,
``test_an_order_picked_while_a_cluster_ranks_does_not_bring_back_the_last_ones`` and, for the
window, ``tests/gui/test_cluster_window.py``'s
``test_an_order_picked_while_the_window_ranks_draws_nothing_until_the_clips_land``.
"""

import math

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QSettings, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from detail.gui.cluster_strip import cluster_colors
from detail.gui.cluster_workspace import TEMPORAL_TERMS, ClusterWorkspace
from detail.gui.label_panel import label_rows


def _drain(qapp, predicate, timeout_ms: int = 10_000) -> bool:
    """Spin the event loop until ``predicate`` holds; every load here is off-thread."""
    from tests.gui.conftest import drain_until

    return drain_until(qapp, predicate, timeout_ms)


def _settled(qapp, measure, quiet_ms: int = 300, timeout_ms: int = 5_000) -> bool:
    """Pump until ``measure()`` has not changed for ``quiet_ms``; False if it never stops."""
    from PyQt6.QtTest import QTest

    last, still, waited = measure(), 0, 0
    while waited < timeout_ms:
        qapp.processEvents()
        QTest.qWait(10)
        waited += 10
        now = measure()
        still = still + 10 if now == last else 0
        last = now
        if still >= quiet_ms:
            return True
    return False


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    """Turn a modal dialog nobody stubbed into a failure instead of a hung run."""
    raised: list[str] = []

    def critical(parent, title, text, *args, **kwargs):
        raised.append(f"{title}: {text}")
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "critical", staticmethod(critical))
    yield
    assert not raised, f"a background call failed: {raised}"


def _workspace(qapp, fitted, terms=None, handle=None):
    experiment, run_id = fitted
    handle = handle or experiment.authored_run(run_id)
    workspace = (
        ClusterWorkspace(experiment, handle, terms)
        if terms is not None
        else ClusterWorkspace(experiment, handle)
    )
    # Every tab loads off-thread and a tab's controls are only right once its own load lands;
    # waiting for INTERPRET alone let a walk over all three read BROWSE before it had delivered.
    assert _drain(
        qapp,
        lambda: workspace._interpret._view is not None
        and workspace._browse._loaded
        and workspace._audit._loaded,
    ), "a tab never loaded"
    # A bout method ranks a node's clips off the UI thread on first ask
    # (`NodeView.ranks_on_demand`), so the opening node's montage lands after the tabs do.
    assert _drain(
        qapp, lambda: workspace._interpret._clips is not None
    ), "the opening node's clips never arrived"
    return experiment, handle, workspace


def _press(interpret, key) -> None:
    interpret.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    )


def _label(qapp, interpret, node: int, name: str, handle) -> None:
    """Label ``node`` through the label field and wait for it to reach the run."""
    interpret.select(node)
    interpret._label_field.edit.setText(name)
    interpret._label_field.submit()
    assert _drain(
        qapp, lambda: handle.state().node_names.get(str(node)) == name
    ), f"{name!r} never reached the run"
    assert _drain(qapp, lambda: not interpret.writes_pending())


def _select(qapp, interpret, node: int) -> None:
    """Select ``node`` and wait for its clips to be showing.

    A bout method ranks a node's clips off the UI thread on first ask
    (``NodeView.ranks_on_demand``), and until they land the montage is empty and the label bar
    says so. A test that reads the node's clips straight after selecting it uses this.
    """
    interpret.select(node)
    assert _drain(
        qapp, lambda: not interpret._clip_text.startswith("ranking")
    ), f"node {node}'s clips never arrived"


def _label_rows(interpret):
    return label_rows(interpret._view, interpret._names)


# --- opening ------------------------------------------------------------------------------


def test_a_fitted_run_opens_on_its_whole_tree(qapp, fitted):
    """No granularity to choose first: the root is selected and every clip of it is listed."""
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    assert interpret._selected == view.root
    assert interpret._icicle.zoom_root == view.root
    assert "100.0% of the run" in interpret.label_bar().figures.text()
    assert len(interpret.montage_order()) == len(view.clips(view.root)) > 0
    assert interpret._key_row.swatches.texts() == [], "nothing labelled yet"


def test_every_block_the_tree_draws_is_a_target_that_selects_its_node(qapp, fitted):
    """The recorded class *a surface that draws N targets and wires the handler for a subset*.

    The pedigree this replaced drew every node and handled clicks on 4 of 48. So: lay the icicle
    out at a real size, then click the centre of **every** block it drew and assert that block's
    node is the one selected. Asserted as a pairing, never as "something was drawn".
    """
    from PyQt6.QtCore import QPoint
    from PyQt6.QtTest import QTest

    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    icicle = interpret._icicle
    workspace.resize(1500, 900)
    workspace.show()
    try:
        qapp.processEvents()
        blocks = icicle._layout()
        assert (
            len(blocks) > 3
        ), "the tree drew almost nothing; the walk would be vacuous"
        for rect, node in blocks:
            centre = rect.center()
            QTest.mouseClick(
                icicle,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                QPoint(int(centre.x()), int(centre.y())),
            )
            assert (
                interpret._selected == node
            ), f"block for {node} selected {interpret._selected}"
    finally:
        workspace.hide()


def test_selecting_a_node_reads_no_video(qapp, fitted, monkeypatch):
    """§22: a click reads the cache and nothing else, so it can never wait on a decode."""
    from detail.api import run_preview

    def refuse(*args, **kwargs):
        raise AssertionError("selecting a node opened a video")

    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    # After the workspace has loaded: BROWSE and AUDIT hold players of their own, and what this
    # asserts is the selection path, not the tabs beside it.
    assert _drain(qapp, lambda: workspace._browse._loaded)
    monkeypatch.setattr(run_preview.ClipQueryPreview, "__init__", refuse)
    for node in range(interpret._view.tree.n_nodes):
        interpret.select(node)
    qapp.processEvents()


def test_the_arrow_keys_walk_the_tree(qapp, fitted):
    """The keys walk the tree **as drawn**: the row under a block, largest first."""
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    row = interpret._icicle.shown_children(view.root)
    assert len(row) >= 2, "the fixture must draw a row under the root"

    _press(interpret, Qt.Key.Key_Down)
    assert interpret._selected == row[0]
    assert view.summary(row[0]).time_ns == max(
        view.summary(c).time_ns for c in row
    ), "down goes to the largest block in the row"
    _press(interpret, Qt.Key.Key_Right)
    assert interpret._selected == row[1]
    _press(interpret, Qt.Key.Key_Left)
    assert interpret._selected == row[0]
    _press(interpret, Qt.Key.Key_Up)
    assert interpret._selected == view.root


class _Job:
    """A stand-in build that never reports, enough for the page to attach to."""

    permille, status = 0, "starting"

    class _Signal:
        def connect(self, _slot):
            pass

    progress = finished = failed = _Signal()


# --- labelling ----------------------------------------------------------------------------


def test_a_label_shows_at_once_and_reaches_the_run(qapp, fitted):
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    node = view.children(view.root)[0]
    interpret.select(node)
    interpret._label_field.edit.setText("moving")
    interpret._label_field.submit()
    # Before any write has landed: the picture follows the label immediately.
    assert interpret._icicle._fills[node][1] is True
    assert any(row[0] == "moving" for row in _label_rows(interpret))
    assert _drain(qapp, lambda: handle.state().node_names.get(str(node)) == "moving")
    # Read the output only once the background write has finished, as BROWSE does (it reloads
    # on `changed`). Polling it mid-write opens `instances.csv` while it is being replaced,
    # which Windows refuses with a sharing violation.
    assert _drain(qapp, lambda: not interpret.writes_pending())
    assert any(
        i.label == "moving" for i in handle.instances()
    ), "the output never caught up"


def test_a_label_inside_a_label_nests_and_the_parent_keeps_an_other_row(qapp, fitted):
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    parent = next(n for n in view.children(view.root) if view.children(n))
    child, sibling = view.children(parent)
    _label(qapp, interpret, parent, "moving", handle)
    _label(qapp, interpret, child, "walking", handle)
    rows = {(text, depth) for text, depth, _root, _time in _label_rows(interpret)}
    assert ("moving", 0) in rows and ("walking", 1) in rows and ("other", 1) in rows
    assert interpret._icicle._fills[child][1] is True
    assert interpret._icicle._fills[sibling][1] is False, "covered by the parent, pale"
    paths = {i.label_path for i in handle.instances() if i.label}
    assert "moving/walking" in paths


def test_a_label_with_the_path_separator_is_refused_before_it_is_shown(
    qapp, fitted, monkeypatch
):
    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[2]))
    )
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    interpret._label(interpret._view.root, "a/b")
    assert warned and "/" in warned[0]
    assert not interpret._names and not handle.state().node_names


def test_removing_a_label_with_u(qapp, fitted):
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    node = interpret._view.children(interpret._view.root)[0]
    _label(qapp, interpret, node, "moving", handle)
    _press(interpret, Qt.Key.Key_U)
    assert _drain(qapp, lambda: str(node) not in handle.state().node_names)


# --- BROWSE ---------------------------------------------------------------------------------
# The tab's own behaviour is `tests/gui/test_label_browse.py`; these are its life in the workspace.


def test_browse_follows_a_label(qapp, fitted):
    """Empty until something is labelled — the unlabelled remainder is not the dataset — and
    the label appears the moment the output catches up."""
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret, browse = workspace._interpret, workspace._browse
    assert _drain(qapp, lambda: browse._loaded)
    assert browse._model.rowCount() == 0
    assert browse._pages.currentIndex() == 0, "empty viewers instead of one message"

    def shows(name) -> bool:
        return any(
            browse._model.item(row, 0).text() == name
            for row in range(browse._model.rowCount())
        )

    view = interpret._view
    _label(qapp, interpret, view.children(view.root)[0], "rearing", handle)
    assert _drain(qapp, lambda: shows("rearing")), "browse never picked up the label"


def test_browse_shows_twenty_rows_with_the_player_level_beside_them(
    qapp, fitted, monkeypatch
):
    from detail.gui.tables import BROWSE_ROWS
    from tests.gui.layout_checks import assert_player_level_with_its_list
    from tests.gui.test_label_browse import labelled_output

    experiment, run_id = fitted
    handle = experiment.authored_run(run_id)
    recording_id = experiment.recordings[0].id
    output = labelled_output(recording_id, 30)
    monkeypatch.setattr(handle, "artifact", lambda: output)
    workspace = ClusterWorkspace(experiment, handle)
    browse = workspace._browse
    workspace.show_view("Browse")
    workspace.resize(1400, 1200)
    workspace.show()
    try:
        assert _drain(qapp, lambda: browse._model.rowCount() == 30)
        qapp.processEvents()
        assert_player_level_with_its_list(
            browse,
            browse._table,
            browse._player._src,
            above=browse._ethogram_band,
            max_rows=BROWSE_ROWS,
        )
        assert browse._table.verticalScrollBar().maximum() > 0, "the rest must scroll"
    finally:
        workspace.hide()


def test_browse_lists_recordings_by_name(qapp, fitted, monkeypatch):
    """§11, and read off the UI thread with the instances, not per row."""
    from tests.gui.test_label_browse import labelled_output

    experiment, handle, workspace = _workspace(qapp, fitted)
    browse = workspace._browse
    recording_id = experiment.recordings[0].id
    monkeypatch.setattr(handle, "artifact", lambda: labelled_output(recording_id, 3))
    payload = browse._load()
    assert payload["names"] == experiment.recording_names()
    browse._show({**payload, "names": {recording_id: "Fish tank A"}})
    shown = {browse._model.item(r, 1).text() for r in range(browse._model.rowCount())}
    assert shown == {"Fish tank A"}, shown


def test_an_export_with_nothing_selected_is_everything_not_the_clip_on_screen(
    qapp, fitted, monkeypatch
):
    """`export.scope_of` kept searching inside a workspace that had answered `None`, and the
    next surface it met was a clip player — so with nothing selected, the shell's Export
    narrowed to whatever clip was playing. A surface's `None` now covers its whole subtree.
    """
    from detail.gui.behavior_query import ClipSpec
    from detail.gui.export import scope_of
    from tests.gui.test_label_browse import labelled_output

    experiment, run_id = fitted
    handle = experiment.authored_run(run_id)
    recording_id = experiment.recordings[0].id
    monkeypatch.setattr(handle, "artifact", lambda: labelled_output(recording_id, 5))
    workspace = ClusterWorkspace(experiment, handle)
    browse = workspace._browse
    workspace.show_view("Browse")
    assert _drain(qapp, lambda: browse._model.rowCount() == 5)
    browse._player.set_clip(
        ClipSpec(
            recording_id=recording_id,
            track_id=0,
            embed_run_id="",
            start_t_ns=0,
            end_t_ns=1_000_000_000,
            label="x",
            score=None,
        )
    )
    page = QWidget()
    QVBoxLayout(page).addWidget(workspace)
    assert scope_of(page) is None, "the clip on screen answered for the workspace"
    browse._table.selectRow(0)
    assert scope_of(page).instance_ids == (browse._rows[0].instance_id,)


# --- finishing, and the way a user reaches it ---------------------------------------------


def _edit_controls_live(interpret) -> dict[str, bool]:
    field = interpret._label_field
    return {
        "label": field.edit.isEnabled(),
        "label button": field.label_button.isEnabled(),
        "rebuild": interpret._rebuild.isEnabled(),
    }


def test_finish_then_reopen_gives_the_label_controls_back(qapp, fitted):
    """§21: finish is a statement, not a lock; Reopen on the same screen brings edits back.

    Rebuild stays live on a finished run on purpose: rebuilding the cache changes what the
    cards can show, never what the run says.
    """
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    assert all(_edit_controls_live(interpret).values())
    interpret._finish_run()
    assert _drain(qapp, lambda: interpret._reopen.isEnabled())
    assert handle.status == "complete"
    live = _edit_controls_live(interpret)
    assert not live["label"] and not live["label button"]
    assert live["rebuild"], "a finished run's cache can still be rebuilt"
    interpret._reopen_run()
    assert handle.status == "open"
    assert all(_edit_controls_live(interpret).values())


def test_the_run_page_opens_the_workspace_while_the_run_is_open(qapp, fitted):
    from detail.gui.pages import AnalysisRunDetailPage

    experiment, run_id = fitted
    page = AnalysisRunDetailPage(experiment, run_id)
    workspaces = page.findChildren(ClusterWorkspace)
    assert workspaces, "the run's page does not open the interpretation workspace"
    assert workspaces[0].view_names() == [
        "Interpret",
        "Audit",
        "Browse",
        "Measures",
    ]
    assert page.view_names() == workspaces[0].view_names()
    assert _drain(qapp, lambda: workspaces[0]._interpret._view is not None)


def test_a_run_whose_job_failed_opens_nothing_and_writes_nothing(qapp, fitted):
    """A failed fit can read ``open``; opening its page must not write to it."""
    import json

    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as analysis_store

    experiment, run_id = fitted
    run_dir = analysis_store.analysis_dir(experiment.dir, run_id)
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "job": {"job_id": run_id, "config": {}},
                "progress": None,
                "error": "PermissionError: [WinError 32]",
            }
        ),
        encoding="utf-8",
    )

    def fingerprint():
        return {
            p.relative_to(run_dir).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(run_dir.rglob("*"))
            if p.is_file()
        }

    before = fingerprint()
    page = AnalysisRunDetailPage(experiment, run_id)
    _drain(qapp, lambda: False, timeout_ms=1_500)
    after = fingerprint()
    assert set(after) == set(before)
    assert {n for n in before if before[n] != after[n]} <= {"analysis.yaml"}
    assert not page.findChildren(ClusterWorkspace)


def test_a_run_with_no_cache_offers_to_build_one(qapp, fitted):
    """A run fitted before the cache existed opens to one action, not a blank screen."""
    import shutil

    from detail.storage import node_cache as node_cache_store

    experiment, run_id = fitted
    handle = experiment.authored_run(run_id)
    shutil.rmtree(node_cache_store.cache_path(handle._run_dir()))
    workspace = ClusterWorkspace(experiment, handle)
    interpret = workspace._interpret
    assert _drain(qapp, lambda: interpret._state is not None)
    assert interpret._view is None
    assert not interpret._missing.isHidden() and interpret._split.isHidden()


def test_building_a_missing_cache_rebuilds_from_the_fit(qapp, fitted, monkeypatch):
    """With no cache, Build asks the run to rebuild it: no floor to choose, nothing else."""
    import shutil

    from detail.gui import node_cache_jobs
    from detail.storage import node_cache as node_cache_store

    experiment, run_id = fitted
    handle = experiment.authored_run(run_id)
    shutil.rmtree(node_cache_store.cache_path(handle._run_dir()))
    asked = []
    monkeypatch.setattr(
        type(handle),
        "rebuild_node_cache",
        lambda self, progress=None: asked.append(progress is not None),
    )

    def start(run_dir, work):
        work(lambda *a: None)
        return _Job()

    monkeypatch.setattr(node_cache_jobs, "start", start)
    workspace = ClusterWorkspace(experiment, handle)
    interpret = workspace._interpret
    assert _drain(qapp, lambda: interpret._state is not None)
    interpret._build.click()
    assert asked == [True], "one rebuild, reporting its progress"


def test_rebuild_rebuilds_and_reloads(qapp, fitted, monkeypatch):
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    old = interpret._view
    asked = []
    real = handle.rebuild_node_cache

    def recording(progress=None):
        asked.append(progress is not None)
        return real(progress=progress)

    monkeypatch.setattr(handle, "rebuild_node_cache", recording)
    interpret._rebuild_cache()
    assert _drain(
        qapp,
        lambda: bool(asked)
        and interpret._view is not None
        and interpret._view is not old,
    )
    assert asked == [True]


# --- generic over the run, not over the method ---------------------------------------------


def test_the_words_come_from_the_terms_not_from_the_module(qapp, fitted):
    _experiment, _handle, workspace = _workspace(qapp, fitted, terms=TEMPORAL_TERMS)
    interpret = workspace._interpret
    words = " ".join(label.text() for label in interpret.findChildren(QLabel))
    words += " " + interpret._label_field.edit.placeholderText()
    assert "behavior" in words
    assert "cluster" not in words.lower(), "the temporal screen speaks behaviors"


def test_colors_stay_distinct_past_the_palette(qapp):
    assert len(set(cluster_colors(8))) == 8
    assert len(set(cluster_colors(60))) == 60


def test_the_workspace_asks_for_the_space_it_needs(qapp, fitted):
    from PyQt6.QtWidgets import QSizePolicy

    _experiment, _handle, workspace = _workspace(qapp, fitted)
    assert workspace.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
    assert workspace.minimumSizeHint().height() >= 500


def test_the_explanatory_text_is_selectable(qapp, fitted):
    """Disabled labels cannot be selected for copying; none of the card's text is disabled."""
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    for label in workspace._interpret.findChildren(QLabel):
        assert label.isEnabled(), label.text()


# --- a snapshot through the same workspace (§21) -------------------------------------------


def _snapshot_workspace(qapp, fitted):
    """A workspace on a pinned snapshot, with the live run **already relabelled past it**."""
    experiment, run_id = fitted
    handle = experiment.authored_run(run_id)
    view = handle.node_view()
    first, second = view.children(view.root)
    handle.name_nodes({first: "pinned"})
    handle.audit_sample(budget=2)
    snapshot_id = handle.pin(reason="finish")
    assert snapshot_id
    handle.name_nodes({first: "", second: "live"})
    _experiment, _view_handle, workspace = _workspace(
        qapp, fitted, handle=handle.view(snapshot_id)
    )
    return handle, workspace, first


def test_a_snapshot_renders_its_own_labels(qapp, fitted):
    handle, workspace, first = _snapshot_workspace(qapp, fitted)
    interpret = workspace._interpret
    assert interpret._names == {str(first): "pinned"}
    assert "live" in handle.state().node_names.values()


#: Controls that stay live on a snapshot, keyed by ``(class, label)``, with the reason.
_LIVE_ON_A_SNAPSHOT = {
    ("QPushButton", "Back to live run"): "the way out",
    # Find clusters rings and zooms the tree and writes nothing, so it is looking. Keyed on the
    # object names (`gui/cluster_find.py`): the field is otherwise ("QLineEdit", ""), which is
    # the label field's entry in the walk's own failure check.
    ("QLineEdit", "find_clusters_field"): "finding a cluster is looking at it",
    ("QPushButton", "find_clusters_find"): "finding a cluster is looking at it",
    ("QPushButton", "find_clusters_chip"): "opening a found cluster is selecting it",
    ("QPushButton", "find_clusters_clear"): "clearing the list is looking",
    # Zooming a frame changes what is on screen, never what the snapshot *is* — the same
    # reason the scrubber's transport controls are here. It appears only once a view has been
    # zoomed, so a frozen page that nobody has zoomed does not show it at all.
    ("QPushButton", "Fit"): "un-zooming a picture is looking at it",
    (
        "QPushButton",
        "Draw all measurements",
    ): "drawing more measurements is looking at them",
    (
        "QPushButton",
        "Draw only the distinctive ones",
    ): "drawing fewer measurements is looking at them",
    (
        "QPushButton",
        "All measurements",
    ): "going back from one measurement to the list",
    # Measuring reads the run and writes nothing it states. The one write it can cause is a pose
    # run's deferred frame views being brought current, which §21 has every reader do.
    ("QPushButton", "Measure this run"): "measuring a run is reading it",
    ("QPushButton", "Re-measure"): "measuring a run again is reading it again",
    # The clip player delegates to the shared scrubber now, so its play control is the
    # scrubber's and carries that widget's object name rather than the text "Play".
    ("QPushButton", "scrubber_play"): "playing a clip is looking at it",
    # The scrubber's own slider: a subclass now, because it draws the clip's span and a pending
    # start on its groove. Scrubbing is looking, as it always was under the plain class name.
    ("_MarkedSlider", ""): "scrubbing a clip is looking at it",
    # Keyed on the widget's object name, not its label: these two were entered as "◀"/"▶" and
    # the entries died the day the buttons were reworded to say what they do.
    (
        "QPushButton",
        "scrubber_prev_frame",
    ): "stepping between frames is looking at them",
    (
        "QPushButton",
        "scrubber_next_frame",
    ): "stepping between frames is looking at them",
    # AUDIT's review pane on a snapshot: moving to the next cluster changes what is on screen;
    # a click on a clip, Move to…, Undo and Finish, which write, stay disabled with the reason.
    (
        "QPushButton",
        "Next cluster",
    ): "opening another cluster's clips is looking at them",
    # A cluster's window plays its clips beside the page; its label row is greyed on a
    # snapshot with the page's reason (`test_cluster_window.py`).
    ("QPushButton", "Open in window"): "opening a cluster's clips is looking at them",
    ("QPushButton", "◀ back (←)"): "stepping through the audit queue is reading it",
    ("QPushButton", "next (→) ▶"): "stepping through the audit queue is reading it",
    ("QPushButton", "▲ load more confident"): (
        "paging the strip — the bare ▲/▼ are the cutoff stepper and stay disabled, which is why "
        "the labels differ"
    ),
    ("QPushButton", "▼ load less confident"): "paging the strip",
    ("QSlider", ""): (
        "thumbnail size and column count — view preferences the surface cannot pick for you "
        "(the granularity slider is a QSlider too, and is disabled, so it never appears here)"
    ),
    ("QPushButton", "▼  Clips"): (
        "folding a band is looking, not editing — the same category as rotating the scatter, "
        "which this file's own docstring calls out as not an edit"
    ),
    ("QPushButton", "▼  Ethograms"): "folding a band is looking, not editing",
    ("QPushButton", "▼  Locations"): "folding a band is looking, not editing",
    # The folded forms, for a snapshot opened after the band was last left shut: the label
    # carries the glyph, so the two states are two different keys.
    #
    # Six entries, not eight: the ethogram band used to title itself from `terms.unit`, so the
    # same band needed a "Cluster ethograms" key *and* a "Behavior ethograms" one. A header
    # shared by both pages is a header that cannot say which page it is on, which is the point.
    ("QPushButton", "▶  Clips"): "folding a band is looking, not editing",
    ("QPushButton", "▶  Ethograms"): "folding a band is looking, not editing",
    ("QPushButton", "▶  Locations"): "folding a band is looking, not editing",
    ("QPushButton", "← All recordings"): (
        "leaving one recording's map for the row of them is navigating between two views of "
        "the same frozen numbers — the same category as folding a band"
    ),
    ("QPushButton", "▼  Interpretation"): "folding a band is looking, not editing",
    ("QPushButton", "▶  Interpretation"): "folding a band is looking, not editing",
    ("QPushButton", "Fit view"): (
        "framing the camera is looking, not editing — the same category as the rotate it "
        "undoes"
    ),
    # The node-centric INTERPRET tab (§22).
    ("QPushButton", "Back to live run"): "the way out",
    # Filtering shades the tree and greys clips; it writes nothing (§22).
    (
        "QPushButton",
        "◂ Hide filters",
    ): "folding the filter dock is looking, not editing",
    ("QPushButton", "Filters ▸"): "opening the filter dock is looking, not editing",
    ("QPushButton", "Reset all"): "resetting the filters is looking, not editing",
    ("QPushButton", "Reset"): (
        "resetting a picked region, or one filter field, is looking, not editing"
    ),
    ("QPushButton", "only"): "ticking one value of a filter is looking, not editing",
    ("QLineEdit", "filter_search"): "searching a filter's values is looking",
    ("QLineEdit", "field_search"): "finding a filter field by name is looking",
    # The montage's order: the same clips, sorted another way.
    ("QPushButton", "montage_order"): "ordering the montage is looking at it",
    (
        "ShowButton",
        "montage_show",
    ): "what the montage's tiles show, and which way they face, is a view choice",
    # Check boxes are walked since the montage's Masks button became one; the clip player's own
    # Masks box was enabled on snapshots the whole time, unwalked.
    ("QCheckBox", "Masks"): "drawing the masks on a clip or not is a view choice",
    # A QPushButton now (it was a QToolButton, smaller than its neighbours, and not walked):
    # its list goes to a label, and its Rename / Merge entries are greyed on a snapshot.
    ("QPushButton", "Labels ▾"): "listing the labels and going to one is looking",
    ("QPushButton", "Build previews"): (
        "building display previews writes to this machine's cache, never to the run"
    ),
    ("QPushButton", "+"): "zooming the tree is looking at it",
    ("QPushButton", "−"): "zooming the tree is looking at it",
    ("QPushButton", "Whole tree"): "un-zooming the tree is looking at it",
    ("QPushButton", "scrubber_play"): "playing a clip is looking at it",
    # The scrubber's own slider: a subclass now, because it draws the clip's span and a pending
    # start on its groove. Scrubbing is looking, as it always was under the plain class name.
    ("_MarkedSlider", ""): "scrubbing a clip is looking at it",
    ("QPushButton", "scrubber_prev_frame"): "stepping between frames is looking",
    ("QPushButton", "scrubber_next_frame"): "stepping between frames is looking",
    # AUDIT's review pane on a snapshot: moving to the next cluster changes what is on screen;
    # a click on a clip, Move to…, Undo and Finish, which write, stay disabled with the reason.
    (
        "QPushButton",
        "Next cluster",
    ): "opening another cluster's clips is looking at them",
    ("QPushButton", "◀ back (←)"): "stepping through the audit queue is reading it",
    ("QPushButton", "next (→) ▶"): "stepping through the audit queue is reading it",
    ("QSlider", ""): "thumbnail size: a view preference",
    ("QPushButton", "Fit"): "un-zooming a picture is looking at it",
    (
        "_FrameEntry",
        "",
    ): "typing a frame number into BROWSE's player is moving through a clip",
    (
        "QPushButton",
        "▼  Ethogram — labelled time, click a bout to play it",
    ): "folding a band is looking, not editing",
    (
        "QLineEdit",
        "browse_recording_search",
    ): "narrowing BROWSE's recording list is looking, not editing",
    (
        "QPushButton",
        "Clear filters",
    ): "widening BROWSE's filters is looking, not editing",
    # AUDIT's navigation: the table, the cleaning scroll, a draw's result, and INTERPRET — each
    # move reads. Every edit on the scroll (Take out, the move buttons, Put back, Estimate
    # precision) stays greyed with the reason.
    ("QPushButton", "Review clips"): "watching a label's clips is looking at them",
    ("QPushButton", "◀ All labels"): "going back to the table is looking",
    (
        "QPushButton",
        "◀ Back to cleaning (Esc)",
    ): "leaving a draw's result for the scroll is looking",
    ("QPushButton", "Least typical first"): "reordering the scroll is looking",
    ("QPushButton", "Most typical first"): "reordering the scroll is looking",
    (
        "QPushButton",
        "Open in INTERPRET",
    ): "opening a cluster on INTERPRET is looking at it",
    (
        "QPushButton",
        "Look at its halves",
    ): "zooming onto an audited cluster's halves is looking at them",
    (
        "FloatSpin",
        "audit_goal_width",
    ): "the goal width is a reading setting: it says when an interval is narrow enough",
    # BROWSE opens on its first instance, so its player has a clip on a snapshot too. Writing
    # that clip to a file reads the snapshot; it changes nothing the snapshot states.
    ("QPushButton", "Export video…"): "writing out the clip on screen is reading it",
    # Every clip grid's Export ▾ and its pick mode (`montage_export`): writing the clips a grid
    # shows reads them, and a pick changes nothing in the run.
    ("ExportMenu", "montage_export"): "writing out a grid's clips is reading them",
    ("QPushButton", "Quit"): "leaving pick mode is looking",
    ("QPushButton", "Pick all"): "picking clips to export is looking",
    (
        "QLineEdit",
        "ethogram_video_picker",
    ): "finding a video on the ethogram scrolls to it; it writes nothing",
    ("QPushButton", "ethogram_whole"): "zooming the ethogram back out is looking at it",
    # The Differences line's Strictness slider reads a saved screen looser or stricter (§22);
    # the screen, the run and the snapshot do not move.
    (
        "QSlider",
        "screen_strictness",
    ): "reading a saved screen at another strictness is looking at it",
}


def _enabled_controls(widget):
    from PyQt6.QtWidgets import (
        QAbstractSpinBox,
        QCheckBox,
        QLineEdit,
        QPushButton,
        QSlider,
    )

    out = []
    for kind in (QPushButton, QCheckBox, QSlider, QAbstractSpinBox, QLineEdit):
        for control in widget.findChildren(kind):
            if not control.isEnabled():
                continue
            if isinstance(control, QLineEdit) and (
                control.isReadOnly() or isinstance(control.parent(), QAbstractSpinBox)
            ):
                continue
            label = control.objectName() or (
                control.text() if isinstance(control, (QPushButton, QCheckBox)) else ""
            )
            out.append((type(control).__name__, label))
    return out


def test_no_control_that_changes_a_snapshot_is_live(qapp, fitted):
    """Walk every control: anything enabled on a snapshot must be on the allowlist above."""
    _handle, workspace, first = _snapshot_workspace(qapp, fitted)
    workspace._interpret.select(first)
    unexpected = [
        pair for pair in _enabled_controls(workspace) if pair not in _LIVE_ON_A_SNAPSHOT
    ]
    assert not unexpected, f"enabled on a snapshot: {unexpected}"


def test_no_control_that_changes_a_snapshot_is_live_before_the_loads_land(qapp, fitted):
    """The same walk, before any tab's background load has delivered.

    Each tab corrected its controls only when its own load landed, so a walk that ran first
    found BROWSE's export buttons live on a snapshot — 20 failures in a week of runs, at no
    particular load. A control has to be right from construction, not from delivery.
    """
    experiment, run_id = fitted
    handle = experiment.authored_run(run_id)
    view = handle.node_view()
    first, _second = view.children(view.root)
    handle.name_nodes({first: "pinned"})
    snapshot_id = handle.pin(reason="finish")
    workspace = ClusterWorkspace(experiment, handle.view(snapshot_id))
    unexpected = [
        pair for pair in _enabled_controls(workspace) if pair not in _LIVE_ON_A_SNAPSHOT
    ]
    assert not unexpected, f"enabled on a snapshot before its load: {unexpected}"


def test_the_walk_would_catch_a_live_control(qapp, fitted):
    """The guard can fail: enable the label field and the walk reports it."""
    _handle, workspace, _first = _snapshot_workspace(qapp, fitted)
    workspace._interpret._label_field.edit.setEnabled(True)
    assert ("QLineEdit", "") in _enabled_controls(workspace)


def test_a_snapshots_audit_can_be_read_but_not_added_to(qapp, fitted):
    """The review pane still shows every clip (watching changes nothing); every edit is dead."""
    from detail.gui.label_check import FROZEN
    from detail.storage import behavior_audit as audit_store

    handle, workspace, _first = _snapshot_workspace(qapp, fitted)
    audit = workspace._audit
    assert _drain(qapp, lambda: audit._loaded), "the audit never loaded"
    assert audit._pages.currentIndex() == audit.PAGE_REVIEW
    before = len(audit_store.read_audit(handle._run_dir()))
    assert audit._open is not None, "the first label's first cluster opens"
    view = audit.pane
    # A bout method ranks the opened cluster's clips off the UI thread; the pane says so until
    # they land (`tests/gui/test_label_check.py::_ranked`).
    assert _drain(qapp, lambda: not view._waiting, 20_000), "the clips never arrived"
    assert not view.finish.isEnabled() and view.finish.toolTip() == FROZEN
    assert not view.montage.is_clickable()
    view.montage.clip_clicked.emit(0, False)
    audit._take_out([0], None)
    audit._finish()
    assert not audit.writes_pending()
    assert len(audit_store.read_audit(handle._run_dir())) == before


def test_leaving_the_page_mid_build_and_coming_back_follows_the_same_build(
    qapp, fitted, monkeypatch
):
    """Reported: building the cache for an existing run "has issues when I leave the page and
    come back". The build ran on a worker owned by the tab, so a page opened later knew nothing
    of it and offered **Build** again, which queued a second build behind the first one's lock.

    Now a build belongs to the application (``gui/node_cache_jobs.py``): tear the first page down
    mid-build, open a second, and it must show the build in flight, must not start another, and
    must load the cache once the one build ends.
    """
    import threading

    from detail.api.clustering_nodes import NodeCacheMixin
    from detail.gui import node_cache_jobs

    release = threading.Event()
    builds: list[bool] = []
    real = NodeCacheMixin.rebuild_node_cache

    def held(self, progress=None):
        builds.append(True)
        if progress is not None:
            progress("caching", 0.4, "positions (1 of 2 animals)")
        assert release.wait(30), "the test never released the build"
        return real(self, progress=progress)

    monkeypatch.setattr(NodeCacheMixin, "rebuild_node_cache", held)
    experiment, run_id = fitted
    _experiment, handle, first = _workspace(qapp, fitted)
    first._interpret._rebuild_cache()
    assert _drain(qapp, lambda: builds == [True])
    run_dir = handle._run_dir()
    assert node_cache_jobs.running(run_dir) is not None

    # Leave the page: the shell tears it down with deleteLater, which runs no page code.
    first.deleteLater()
    qapp.processEvents()

    second = ClusterWorkspace(experiment, experiment.authored_run(run_id))
    interpret = second._interpret
    try:
        # At once, before any event is processed: not the empty workspace under a "Loading…"
        # overlay first (reported: "I see the second image for a second before it goes back").
        assert not interpret._building.isHidden()
        assert interpret._split.isHidden()
        assert _drain(qapp, lambda: not interpret._building.isHidden())
        assert interpret._split.isHidden() and interpret._missing.isHidden()
        assert "40%" in interpret._build_status.text(), "it shows where the build is"
        assert not interpret._rebuild.isEnabled()
        interpret._rebuild_cache()  # pressing it again follows the same build
        assert len(builds) == 1, "a second build was started"

        release.set()
        assert _drain(qapp, lambda: node_cache_jobs.running(run_dir) is None, 30_000)
        # The tree comes back once the reload the build's end triggers has landed.
        assert _drain(qapp, lambda: not interpret._split.isHidden())
        assert interpret._building.isHidden() and interpret._view is not None
    finally:
        release.set()
        _drain(qapp, lambda: node_cache_jobs.running(run_dir) is None, 30_000)


def test_a_run_still_working_is_not_opened_for_interpretation(qapp, fitted):
    """Reported: "no behavior_output.yaml" from a new run. Its runner had written ``open``
    before the work that ends the job, and the run page believed the record: it built the
    workspace mid-fit and its tabs read an artifact that was not there yet. While the job
    envelope says the job is still running, the page shows a run in progress instead.
    """
    import json
    import os

    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as analysis_store

    experiment, run_id = fitted
    run_dir = analysis_store.analysis_dir(experiment.dir, run_id)
    assert analysis_store.read_analysis_yaml(run_dir)["status"] == "open"
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "job": {"job_id": run_id, "config": {}},
                "progress": None,
                "error": None,
                # A live pid, so reconciliation reads a working job, not a dead one.
                "pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )
    assert experiment.analysis_job_active(run_id)
    page = AnalysisRunDetailPage(experiment, run_id)
    _drain(qapp, lambda: False, timeout_ms=500)
    assert not page.findChildren(
        ClusterWorkspace
    ), "the workspace opened over a running job"
    assert (
        analysis_store.read_analysis_yaml(run_dir)["status"] == "open"
    ), "display only"


# --- the page ---------------------------------------------------------------------------


def test_a_fresh_page_shows_no_empty_workspace_while_it_loads(qapp, fitted):
    """Under the loading overlay the tree and card were drawn empty, which read as broken."""
    experiment, run_id = fitted
    workspace = ClusterWorkspace(experiment, experiment.authored_run(run_id))
    interpret = workspace._interpret
    assert (
        interpret._split.isHidden()
    ), "nothing drawn before there is something to draw"
    assert _drain(qapp, lambda: interpret._view is not None)
    assert not interpret._split.isHidden()


# --- labelling up front, honest numbers, click-through ------------------------------------


def _settle(qapp, interpret):
    assert _drain(qapp, lambda: not interpret.writes_pending())


def test_typing_an_existing_name_says_it_adds_to_that_label(qapp, fitted):
    """Two clusters given one name are one label (§22), and the box says so before the click:
    the button reads "Add to ‘moving’", not "Label"."""
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    a, b = view.children(view.root)
    _label(qapp, interpret, a, "moving", handle)
    interpret.select(b)
    field = interpret._label_field
    field.edit.setText("Moving")
    assert field.label_button.text() == "Add to “moving”"
    field.edit.setText("resting")
    assert field.label_button.text() == "Label"


# --- the label bar (review round 9, option A) ----------------------------------------------


def test_the_label_bar_sits_between_the_tree_and_the_montage(qapp, fitted):
    """Labelling is the page's main act, so its bar is where the eye goes from the tree to the
    clips, not a row above the tree. Measured on screen, after the layout has run."""
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    workspace.resize(1500, 1000)
    workspace.show()
    try:
        qapp.processEvents()
        main = interpret._main
        bar = interpret.label_bar()

        def top(widget):
            return widget.mapTo(main, widget.rect().topLeft()).y()

        tree_bottom = top(interpret._tree_holder) + interpret._tree_holder.height()
        assert tree_bottom <= top(bar), "the bar is not under the tree"
        assert top(bar) + bar.height() <= top(
            interpret.montage()
        ), "nor above the clips"
        assert bar.isAncestorOf(interpret._label_field), "the field is in the bar"
        assert not isinstance(interpret.filters(), QWidget), "no row above the tree"
    finally:
        workspace.hide()


def test_the_figures_moved_from_the_montage_header_to_the_bar(qapp, fitted):
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    node = view.children(view.root)[0]
    interpret.select(node)
    bar = interpret.label_bar()
    assert bar.name.text() == f"Cluster #{node}"
    assert "% of the run" in bar.figures.text()
    assert bar.isAncestorOf(interpret._audit_block)
    montage_text = " ".join(
        label.text() for label in interpret.montage().findChildren(QLabel)
    )
    assert "of the run" not in montage_text and f"#{node}" not in montage_text
    assert not hasattr(interpret.montage(), "subtitle")


def test_the_chip_shows_own_inherited_and_no_label(qapp, fitted):
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    parent = next(n for n in view.children(view.root) if view.children(n))
    child = view.children(parent)[0]
    chip = interpret.label_bar().chip
    interpret.select(parent)
    assert chip.text() == "not labelled" and chip.colour() is None
    _label(qapp, interpret, parent, "climb", handle)
    assert chip.text() == "climb" and chip.is_filled()
    interpret.select(child)
    assert chip.text() == "inside “climb”" and not chip.is_filled()


def test_a_recent_label_entry_labels_the_open_cluster_through_the_queue(qapp, fitted):
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    a, b = view.children(view.root)
    bar = interpret.label_bar()
    quick = bar.quick
    assert not quick.actions(), "no labels yet, no recent labels"
    _label(qapp, interpret, a, "climb", handle)
    interpret.select(b)
    assert quick.names()[0] == "climb"
    assert not [
        x for x in bar.findChildren(QPushButton) if x.text().startswith("Recent")
    ], "no Recent ▾: the recent labels lead Labels ▾"
    assert quick.action(0).text() == "1  climb"
    labels_menu = bar.field.labels_menu()
    entries = labels_menu.actions()
    assert entries[0].text() == "Recent (keys 1–9)" and not entries[0].isEnabled()
    assert entries[1].text() == "1  climb", "numbered, so the key stays discoverable"
    assert entries[1] is not quick.action(0), "the menu's entry fills; the key writes"
    entries[1].trigger()
    assert bar.field.edit.text() == "climb" and str(b) not in interpret._names
    quick.action(0).trigger()  # what key 1 does
    # Shown at once, written behind it by the page's one write queue.
    assert interpret._names[str(b)] == "climb"
    assert interpret.writes_pending()
    assert interpret.label_bar().status.text().startswith(f"Labelled #{b} “climb”")
    assert _drain(qapp, lambda: handle.state().node_names.get(str(b)) == "climb")
    assert _drain(qapp, lambda: not interpret.writes_pending())
    assert "Output up to date." in interpret.label_bar().status.text()


def _reloads_landed(qapp) -> None:
    """Every off-thread read a label write started (BROWSE's and AUDIT's reloads) delivered."""
    from tests.gui.conftest import drain_busy_calls

    drain_busy_calls(qapp)
    qapp.processEvents()


def test_key_one_labels_with_the_first_recent_label_but_not_while_typing(qapp, fitted):
    """Real keys to whatever holds the focus. It was flaky: the label write's AUDIT reload,
    landing while the test ran, opened a cluster in the hidden AUDIT tab and took the focus
    there, so the field the test had focused no longer had it."""
    from PyQt6.QtTest import QTest

    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    a, b = view.children(view.root)
    workspace.show()
    try:
        _label(qapp, interpret, a, "climb", handle)
        interpret.select(b)
        _reloads_landed(qapp)
        interpret._label_field.edit.setFocus()
        qapp.processEvents()
        assert interpret.focusWidget() is interpret._label_field.edit
        QTest.keyClick(interpret.focusWidget(), Qt.Key.Key_1)
        assert str(b) not in interpret._names, "a digit typed in the field labelled"
        interpret._label_field.edit.clear()
        interpret._icicle.setFocus()
        qapp.processEvents()
        QTest.keyClick(interpret.focusWidget(), Qt.Key.Key_1)
        assert interpret._names.get(str(b)) == "climb"
        assert _drain(qapp, lambda: handle.state().node_names.get(str(b)) == "climb")
        assert _drain(qapp, lambda: not interpret.writes_pending())
    finally:
        workspace.hide()


@pytest.mark.parametrize(
    "modifier",
    [
        Qt.KeyboardModifier.ControlModifier,
        Qt.KeyboardModifier.MetaModifier,
        Qt.KeyboardModifier.AltModifier,
        Qt.KeyboardModifier.ShiftModifier,
    ],
)
def test_a_labelling_key_with_a_modifier_labels_nothing(qapp, fitted, modifier):
    """Review: the page's 1–9 / U / L ignored modifiers, so ⌘3 labelled; as in a cluster window
    they take none (the keypad's excepted). Sent as real keys to the focused tree."""
    from PyQt6.QtTest import QTest

    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    a, b = view.children(view.root)
    _label(qapp, interpret, a, "climb", handle)
    interpret.select(b)
    workspace.show()
    try:
        interpret._icicle.setFocus()
        qapp.processEvents()
        QTest.keyClick(interpret._icicle, Qt.Key.Key_1, modifier)
        assert str(b) not in interpret._names
        interpret.select(a)
        QTest.keyClick(interpret._icicle, Qt.Key.Key_U, modifier)
        assert interpret._names.get(str(a)) == "climb"
        QTest.keyClick(interpret._icicle, Qt.Key.Key_L, modifier)
        assert interpret.focusWidget() is not interpret._label_field.edit
        interpret.select(b)
        QTest.keyClick(
            interpret._icicle, Qt.Key.Key_1, Qt.KeyboardModifier.KeypadModifier
        )
        assert interpret._names.get(str(b)) == "climb", "the keypad's 1 labels"
        assert _drain(qapp, lambda: not interpret.writes_pending())
    finally:
        workspace.hide()


def test_a_label_written_on_interpret_leaves_the_keyboard_on_interpret(qapp, fitted):
    """A label write emits ``changed``, which reloads the hidden AUDIT tab; its reload opened a
    cluster and called ``setFocus`` on AUDIT's grid. Keys 1–9, the arrows, L and U then went
    nowhere on INTERPRET until a click. The focus stays where the user left it."""
    from PyQt6.QtTest import QTest

    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    a, b = view.children(view.root)
    workspace.resize(1400, 900)
    workspace.show()
    try:
        assert workspace._tabs.currentWidget() is interpret
        interpret._icicle.setFocus()
        qapp.processEvents()
        _label(qapp, interpret, a, "climb", handle)
        interpret.select(b)
        interpret._icicle.setFocus()
        qapp.processEvents()
        audit_reads = []
        real = workspace._audit._render
        workspace._audit._render = lambda data, **kw: (
            real(data, **kw),
            audit_reads.append(1),
        )
        workspace._interpret.changed.emit()  # as every label write does
        assert _drain(qapp, lambda: audit_reads), "AUDIT's reload never landed"
        _reloads_landed(qapp)
        focus = workspace.focusWidget()
        assert focus is not None and interpret.isAncestorOf(
            focus
        ), f"the keyboard went to {type(focus).__name__} outside INTERPRET"
        assert workspace._audit.focusWidget() is None
        QTest.keyClick(focus, Qt.Key.Key_1)
        assert interpret._names.get(str(b)) == "climb", "1 labels the open cluster"
        assert _drain(qapp, lambda: handle.state().node_names.get(str(b)) == "climb")
        assert _drain(qapp, lambda: not interpret.writes_pending())
    finally:
        workspace.hide()


def test_a_recent_label_the_cluster_may_not_join_is_greyed_with_the_reason(
    qapp, fitted
):
    """The parent rule, before the click: a label inside “moving” is not offered to a cluster
    at the top level — greyed, saying why, and its number key does nothing."""
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    parent = next(n for n in view.children(view.root) if view.children(n))
    other = next(n for n in view.children(view.root) if n != parent)
    _label(qapp, interpret, parent, "moving", handle)
    _label(qapp, interpret, view.children(parent)[0], "rest", handle)
    interpret.select(other)
    quick = interpret.label_bar().quick
    index = quick.names().index("rest")
    entry = quick.action(index)
    assert not entry.isEnabled()
    assert "share one parent label" in entry.toolTip()
    assert quick.action(quick.names().index("moving")).isEnabled()
    _press(interpret, Qt.Key.Key_1 + index)
    assert str(other) not in interpret._names


def test_recent_labels_lead_with_the_last_used(qapp, fitted):
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    a, b = view.children(view.root)
    _label(qapp, interpret, a, "climb", handle)
    _label(qapp, interpret, b, "rest", handle)
    assert interpret.label_bar().quick.names() == ["rest", "climb"]


def test_rename_and_merge_from_the_labels_panel_reach_the_run(
    qapp, fitted, monkeypatch
):
    """From the label list: Rename… renames every cluster under it, Merge into another… moves
    them all — both written through the run's label operations."""
    from PyQt6.QtWidgets import QInputDialog

    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    a, b = view.children(view.root)
    _label(qapp, interpret, a, "climb", handle)
    _label(qapp, interpret, b, "scale", handle)

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("climbing", True))
    )
    interpret._rename_label("climb")
    assert _drain(qapp, lambda: "climbing" in handle.state().labels.values())
    assert _drain(qapp, lambda: not interpret.writes_pending())

    monkeypatch.setattr(
        QInputDialog, "getItem", staticmethod(lambda *a, **k: ("climbing", True))
    )
    interpret._merge_label("scale")
    assert _drain(qapp, lambda: list(handle.state().labels.values()) == ["climbing"])
    assert _drain(qapp, lambda: not interpret.writes_pending())
    assert handle.state().node_names == {str(a): "climbing", str(b): "climbing"}


def test_double_clicking_a_block_opens_a_label_editor_there(qapp, fitted):
    from PyQt6.QtCore import QPoint

    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    node = view.children(view.root)[0]
    interpret._icicle.label_wanted.emit(node, QPoint(10, 10))
    editor = interpret._inline_editor
    assert editor is not None and editor.isVisible()
    editor.edit.setText("rearing")
    editor.edit.returnPressed.emit()
    assert interpret._names[str(node)] == "rearing"
    assert interpret._selected == node
    _settle(qapp, interpret)


def test_zooming_onto_a_block_is_on_its_menu(qapp, fitted):
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    node = interpret._view.children(interpret._view.root)[0]
    texts = [a.text() for a in interpret._icicle.context_menu(node).actions()]
    assert "Label…" in texts and "Zoom to this" in texts and "Remove label" in texts


def test_cluster_ids_typed_into_find_are_ringed_and_the_tree_zooms_to_hold_them(
    qapp, fitted
):
    """An id ringed, the tree zoomed right onto it, a new search replacing it, the chip
    selecting its cluster, and Clear taking it off."""
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    left, right = view.children(view.root)
    b = view.children(left)[0] if view.children(left) else left
    c = view.children(right)[0] if view.children(right) else right
    find = interpret._find
    find.field.setText(f"#{b}")
    find.submit()
    assert list(interpret._icicle.found()) == [b]
    assert interpret._icicle.zoom_root == b, "one cluster zooms right onto it"
    find.field.setText(f"{c}")
    find.submit()
    assert list(interpret._icicle.found()) == [c], "a new search replaces the last"
    assert interpret._icicle.zoom_root == c
    find.field.setText("999999")
    find.submit()
    assert interpret._icicle.found() == {}, "an id the run lacks rings nothing"
    find.field.setText(f"{b}")
    find.submit()
    find.chip().name.click()
    assert interpret._selected == b
    find.clear.click()
    assert interpret._icicle.found() == {}
    assert (
        interpret._icicle.zoom_root == view.root
    ), "Clear shows the view before finding"


def test_clear_goes_back_to_the_view_before_finding(qapp, fitted):
    """The user's own zoom, not the whole tree: Clear puts back the zoom root, the visible span
    and the level scroll that were on screen when finding began, however the zoom moved and
    however many searches came between (user, 2026-11-16)."""
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    icicle = interpret._icicle
    view = interpret._view
    left, right = view.children(view.root)
    icicle.zoom_to(left)
    icicle.zoom.set_view(0.25, 0.75)
    before = icicle.view_state()
    find = interpret._find
    find.search(right)
    assert icicle.zoom_root == right
    icicle.zoom_by(0.5)  # the user zooms about while finding
    find.search(left)  # and searches again: the view to go back to is still the first
    find.clear.click()
    assert icicle.view_state() == before


def test_find_by_label_rings_every_cluster_with_it_and_a_new_label_follows(
    qapp, fitted
):
    """Typing a label finds the clusters the user gave it, whole-name and ignoring case, and
    zooms to hold them. Labelling one more cluster while it is listed rings that one too and
    leaves the zoom alone: a label landing never moves the tree."""
    experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    icicle = interpret._icicle
    view = interpret._view
    left, right = view.children(view.root)
    a = view.children(left)[0] if view.children(left) else left
    _label(qapp, interpret, a, "climb", handle)
    find = interpret._find
    find.field.setText("CLIMB")
    find.submit()
    assert list(icicle.found()) == [a]
    assert icicle.zoom_root == a
    assert (
        icicle.found()[a].name() != interpret._colour_of(a).name()
    ), "the band is the block's own fill colour, so it does not show"
    _label(qapp, interpret, right, "climb", handle)
    assert set(icicle.found()) == {a, right}, "the new label is ringed too"
    assert icicle.zoom_root == a, "and the tree did not move"
    assert find.chip().name.text() == "CLIMB · 2"


def test_find_shares_the_zoom_row_and_costs_the_tree_no_height(qapp, fitted):
    """Find sits in the zoom row's spare width, not a row of its own above the tree (user,
    2026-11-16): the row it would have added was 31px of a 171px tree pane at 1440x847. Idle,
    the row is exactly as tall as the zoom buttons; listing ids does not move the tree.
    """
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    workspace.resize(1440, 847)
    workspace.show()
    interpret = workspace._interpret
    find = interpret._find
    assert (
        find.parentWidget() is interpret._zoom_in.parentWidget()
    ), "not in the zoom row"
    qapp.processEvents()
    row = find.parentWidget()
    assert row.height() == interpret._zoom_in.height(), "the idle row grew"
    tree_top = interpret._tree.mapTo(workspace, interpret._tree.rect().topLeft()).y()
    view = interpret._view
    find.search(view.children(view.root)[0])
    qapp.processEvents()
    assert (
        interpret._tree.mapTo(workspace, interpret._tree.rect().topLeft()).y()
        == tree_top
    )
    workspace.close()


def test_a_refit_clears_find_because_every_id_names_another_cluster(qapp, fitted):
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    node = view.children(view.root)[0]
    interpret._find.search(node)
    assert interpret._icicle.zoom_root == node
    interpret._tree_key = ("a different fit",)
    interpret._reloaded_tree(view)
    assert interpret._find.entry() is None and interpret._icicle.found() == {}
    assert interpret._icicle.zoom_root == view.root


# --- AUDIT: a check of one label -------------------------------------------------------------
# The tab's own behaviour is `tests/gui/test_label_check.py`; these are its life in the workspace.


def _audit_ready(qapp, workspace):
    from tests.gui.conftest import drain_busy_calls

    audit = workspace._audit
    audit.reload()
    assert _drain(qapp, lambda: audit._loaded)
    drain_busy_calls(qapp)
    return audit


def test_the_audit_opens_on_a_label_with_no_scope_spanning_every_label(qapp, fitted):
    """No "is this behavior?" over every label (#510): the page opens on a label's cluster."""
    from detail.gui.label_check import KEY_ROLE

    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    a, b = view.children(view.root)
    _label(qapp, interpret, a, "climb", handle)
    _label(qapp, interpret, b, "climb", handle)  # one label, two clusters
    audit = _audit_ready(qapp, workspace)
    assert audit._label == "climb"
    assert audit._pages.currentIndex() == audit.PAGE_REVIEW
    assert audit._list.currentItem().data(0, KEY_ROLE)[:2] == ("cluster", "climb")


def test_moving_to_audit_from_a_labelled_cluster_audits_its_label(qapp, fitted):
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    a, b = view.children(view.root)
    _label(qapp, interpret, a, "climb", handle)
    _label(qapp, interpret, b, "scale", handle)
    interpret.select(view.children(b)[0])  # inside "scale": its label covers it
    workspace.show_view("Audit")
    assert _drain(qapp, lambda: workspace._audit._label == "scale")


def test_open_in_interpret_lands_on_the_cluster(qapp, fitted):
    """The audit found a weak cluster; one click puts it in front of the user on INTERPRET."""
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    a, _b = view.children(view.root)
    # Where selecting it directly lands — a node drawn only through its pieces lands on the
    # block that stands for it, and Open in INTERPRET must land in the same place.
    interpret.select(int(a))
    _drain(qapp, lambda: True)
    landing = interpret._selected
    interpret.select(view.root)
    workspace.show_view("Audit")
    workspace._audit.node_requested.emit(int(a))
    assert workspace.current_view() == "Interpret"
    assert workspace._tabs.currentWidget() is interpret
    assert _drain(qapp, lambda: interpret._selected == landing)


def test_the_measures_are_a_tab_of_their_own(qapp, fitted):
    """Not a section under every tab, where under AUDIT it read as part of the review."""
    from detail.gui.pages import AnalysisRunDetailPage, MeasuresPanel

    experiment, run_id = fitted
    page = AnalysisRunDetailPage(experiment, run_id)
    workspace = page.findChildren(ClusterWorkspace)[0]
    measures = next(v.widget for v in workspace.views() if v.name == "Measures")
    assert measures.findChildren(MeasuresPanel)
    assert len(page.findChildren(MeasuresPanel)) == 1, "measures shown twice"


def test_out_of_date_clips_say_so_with_the_way_to_fix_them(qapp, fitted, monkeypatch):
    from detail.api.node_view import NodeView

    monkeypatch.setattr(
        NodeView, "stale_reason", lambda self, current: "the fit size changed"
    )
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    assert not interpret._stale.isHidden()
    assert "Rebuild clips" in interpret._stale.text()


# --- AUDIT's figures on INTERPRET ---------------------------------------------------------


def _audit_row(node: int, label: str = "climb") -> dict:
    """An ``audit_by_cluster`` row: 2 of 8 audited clips were the label, 32 left to audit."""
    return {
        "node": int(node),
        "label": label,
        "path": label,
        "size": 40,
        "share": 1.0,
        "estimate": {
            "n": 8,
            "n_confirmed": 2,
            "precision": 0.25,
            "interval": (0.07, 0.59),
            "size": 40,
            "remaining": 32,
        },
        "answers": {"not climb": 6, "climb": 2},
    }


def _reviewed_row(node: int, label: str = "climb") -> dict:
    """A ``review_by_cluster`` row for a cluster whose every clip is reviewed: 2 of 8 were the
    label. INTERPRET shows a temporal run's cluster a figure only once its review is complete.
    """
    row = _audit_row(node, label)
    return {
        **row,
        "size": 8,
        "estimate": {**row["estimate"], "size": 8, "remaining": 0},
    }


def _stub_review(monkeypatch, rows: list[dict]) -> list[dict]:
    """What INTERPRET reads for a run with clip review (``_read_audit``)."""
    from detail.api.temporal_clustering import TemporalClusteringRun

    monkeypatch.setattr(
        TemporalClusteringRun, "review_by_cluster", lambda self, **_kw: list(rows)
    )
    return rows


def test_the_ring_is_painted_inside_the_block(qapp, fitted):
    from PyQt6.QtGui import QColor

    _e, _h, workspace = _workspace(qapp, fitted)
    icicle = workspace._interpret._icicle
    workspace.resize(1400, 900)
    workspace.show()
    qapp.processEvents()
    icicle.grab()  # lays the blocks out
    rect, node = max(icicle._blocks, key=lambda pair: pair[0].width())
    want = QColor("#ff00ff")
    icicle.set_outlines({node: want})
    image = icicle.grab().toImage()
    ratio = image.devicePixelRatio()
    colour = image.pixelColor(
        int((rect.left() + 3) * ratio), int(rect.center().y() * ratio)
    )
    assert abs(colour.red() - 255) + colour.green() + abs(colour.blue() - 255) < 90


def test_a_judgment_on_audit_rings_the_cluster(qapp, fitted, monkeypatch):
    """A cluster nobody has audited has no ring; the first judgment on AUDIT puts one on it."""
    experiment, run_id = fitted
    handle = experiment.authored_run(run_id)
    view = handle.node_view()
    first, _second = view.children(view.root)
    handle.name_nodes({first: "climb"})
    unaudited = {**_reviewed_row(first), "estimate": {"n": 0}}
    rows = _stub_review(monkeypatch, [unaudited])
    _e, _h, workspace = _workspace(qapp, fitted, handle=handle)
    interpret = workspace._interpret
    assert first not in interpret._icicle.outlines()
    rows[0] = _reviewed_row(first)
    workspace._audit.changed.emit()
    assert _drain(qapp, lambda: first in interpret._icicle.outlines())


def test_an_edit_on_interpret_refreshes_the_ring_and_the_card(
    qapp, fitted, monkeypatch
):
    """Remove from the label moves the cluster out of what was audited; the ring and the card
    kept showing it until the page was reopened."""
    experiment, run_id = fitted
    handle = experiment.authored_run(run_id)
    view = handle.node_view()
    first, _second = view.children(view.root)
    handle.name_nodes({first: "climb"})
    rows = _stub_review(monkeypatch, [_reviewed_row(first)])
    _e, _h, workspace = _workspace(qapp, fitted, handle=handle)
    interpret = workspace._interpret
    interpret.select(first)
    assert first in interpret._icicle.outlines()
    rows.clear()  # the cluster is no longer labelled, so it has no audit row
    interpret._audit_block._remove.trigger()
    assert _drain(qapp, lambda: not handle.state().node_names.get(str(first)))
    assert _drain(qapp, lambda: first not in interpret._icicle.outlines())
    assert interpret._audit_block.isHidden()


def test_an_audit_read_overtaken_by_a_reload_is_dropped(qapp, fitted, monkeypatch):
    """A refresh started before a full reload lands after it: it must not paint over the
    reload's newer reading."""
    experiment, run_id = fitted
    handle = experiment.authored_run(run_id)
    view = handle.node_view()
    first, _second = view.children(view.root)
    handle.name_nodes({first: "climb"})
    _stub_review(monkeypatch, [_reviewed_row(first)])
    _e, _h, workspace = _workspace(qapp, fitted, handle=handle)
    interpret = workspace._interpret
    interpret.refresh_audit()
    # A full reload lands while the refresh is in flight, with its own, newer reading.
    interpret._generation += 1
    newer = {}
    interpret._audited = newer
    assert _drain(qapp, lambda: interpret._audit_reader is None)
    assert interpret._audited is newer, "the older read painted over the reload's"


def test_move_to_greys_a_label_the_cluster_may_not_join(qapp):
    """Move to… offered every label and let the run refuse the move afterwards."""
    from detail.gui.node_audit import NodeAuditBlock

    block = NodeAuditBlock()
    block.show_node(
        3,
        _audit_row(3),
        [("rear", 1, None), ("corner", 1, "Already inside “corner”.")],
        has_halves=False,
        editable_reason="",
    )
    actions = {a.text(): a for a in block._move_menu.actions()}
    assert actions["rear"].isEnabled()
    assert not actions["corner"].isEnabled()
    assert actions["corner"].toolTip() == "Already inside “corner”."


def test_look_at_its_halves_opens_the_first_half(qapp, fitted, monkeypatch):
    experiment, run_id = fitted
    handle = experiment.authored_run(run_id)
    view = handle.node_view()
    first, _second = view.children(view.root)
    handle.name_nodes({first: "climb"})
    _stub_review(monkeypatch, [_reviewed_row(first)])
    _e, _h, workspace = _workspace(qapp, fitted, handle=handle)
    interpret = workspace._interpret
    interpret.select(first)
    interpret._audit_block._halves.trigger()
    assert interpret._selected == view.children(first)[0]
    assert interpret._icicle._root == first


def test_a_snapshot_shows_the_figure_but_not_the_fixes(qapp, fitted, monkeypatch):
    experiment, run_id = fitted
    view = experiment.authored_run(run_id).node_view()
    first, _second = view.children(view.root)
    _stub_review(monkeypatch, [_reviewed_row(first, label="pinned")])
    _handle, workspace, first = _snapshot_workspace(qapp, fitted)
    interpret = workspace._interpret
    interpret.select(first)
    block = interpret._audit_block
    assert not block.isHidden()
    assert not block._move.isEnabled() and not block._remove.isEnabled()
    assert "snapshot" in block._why.text()


@pytest.mark.parametrize("size", [(1440, 760), (1920, 900)])
def test_audit_fits_a_laptop_window_inside_the_run_page(qapp, fitted, size):
    """The reported bug: on a laptop the check ran past the bottom of the run page, which
    scrolled. The review pane is held to the window: the page does not scroll and the montage's
    grid ends on screen."""
    from PyQt6.QtCore import QPoint

    from detail.gui.pages import AnalysisRunDetailPage
    from detail.gui.scroll import PageScroll
    from tests.gui.conftest import drain_busy_calls

    experiment, run_id = fitted
    handle = experiment.authored_run(run_id)
    view = handle.node_view()
    first, second = view.children(view.root)
    handle.name_nodes({first: "climb", second: "rear"})
    scroll = PageScroll()
    run_page = AnalysisRunDetailPage(experiment, run_id)
    workspace = run_page.findChildren(ClusterWorkspace)[0]
    run_page._show_view("Audit")
    scroll.setWidget(run_page)
    scroll.resize(*size)
    scroll.show()
    assert _drain(qapp, lambda: workspace._interpret._view is not None)
    tab = workspace._audit
    assert _drain(qapp, lambda: tab._loaded)
    drain_busy_calls(qapp)
    assert tab._open is not None, "a cluster is open for review"
    bar = scroll.verticalScrollBar()
    assert _settled(qapp, lambda: bar.maximum()), "the layout never settled"
    assert bar.maximum() <= 4, f"the run page scrolls by {bar.maximum()}px"
    grid = tab.pane.montage.grid
    bottom = grid.mapTo(scroll.viewport(), QPoint(0, grid.height())).y()
    assert bottom <= scroll.viewport().height() + 2, "the montage runs off the window"
    assert grid.height() > 200, "the montage was squeezed to nothing"


def test_audit_shows_every_row_after_labelling_on_interpret(qapp, fitted):
    """The reported bug, on the table this list replaced: label clusters on INTERPRET, open AUDIT,
    and rows hid behind a scrollbar in a page with room to spare. The list fills the page's
    height, so every label and cluster row is on screen."""
    from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

    from detail.gui.label_check import LIST_PX
    from detail.gui.scroll import PageScroll
    from tests.gui.conftest import drain_busy_calls

    _experiment, handle, workspace = _workspace(qapp, fitted)
    scroll = PageScroll()
    run_page = QWidget()
    column = QVBoxLayout(run_page)
    header = QLabel("Status: open")
    header.setFixedHeight(100)
    column.addWidget(header)
    column.addWidget(workspace)
    scroll.setWidget(run_page)
    scroll.resize(1440, 900)
    scroll.show()
    interpret = workspace._interpret
    view = interpret._view
    a, b = view.children(view.root)
    a1, a2 = view.children(a)
    for node, name in ((a, "climb"), (a1, "center"), (a2, "left"), (b, "rear")):
        _label(qapp, interpret, node, name, handle)
    drain_busy_calls(qapp)
    workspace.show_view("Audit")
    tab = workspace._audit
    assert _drain(qapp, lambda: tab._loaded)
    drain_busy_calls(qapp)
    for _ in range(10):
        qapp.processEvents()
    lst = tab._list
    assert lst.width() >= LIST_PX, "opens fitted to its widest row, never narrower"
    assert not lst.horizontalScrollBar().isVisible()
    assert not lst.verticalScrollBar().isVisible(), "a short list hid rows behind a bar"
    rows = list(tab._items())
    assert len(rows) == 8, "four labels, each with its cluster"
    for item in rows:
        rect = lst.visualItemRect(item)
        assert rect.bottom() <= lst.viewport().height(), "a row is cut off"


# --- the montage: every clip of the selected cluster ------------------------------------------


def _most_clips(view) -> int:
    """The node with the most clips: the fixture's one recording is one stretch at the root,
    and a cluster deeper down is where it breaks into several."""
    return max(range(view.tree.n_nodes), key=lambda n: len(view.clips(n)))


def test_every_cluster_lists_every_clip_in_the_order_asked(qapp, fitted):
    """No floor: the smallest cluster plays every clip as the root does, most typical first by
    default, and the order switch re-sorts the same clips (``ClipTable.order``, seeded by node).
    """
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    montage = interpret.montage()
    busiest = _most_clips(view)
    assert len(view.clips(busiest)) >= 3, "the fixture must break a cluster into clips"
    for node in (view.root, busiest):
        _select(qapp, interpret, node)
        clips = handle.clips(node)  # as the run ranks them
        assert len(montage._clips) == len(clips) > 0
        assert interpret.montage_order() == clips.order("typical", seed=node).tolist()
        for key in ("atypical", "random"):
            montage.order_actions[key].trigger()
            assert interpret.montage_order() == clips.order(key, seed=node).tolist()
        montage.order_actions["typical"].trigger()
    assert not hasattr(interpret._icicle, "_uncached")


def test_a_cluster_ranked_on_demand_is_ranked_off_the_ui_thread(
    qapp, fitted, monkeypatch
):
    """A medoid method ranks a node from its fit on first ask; the page asks the run on a
    worker and plays the clips in that order when they land."""
    import threading

    import numpy as np

    from detail.api.node_view import NodeView

    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    asked = []

    def ranked(self, node, *, progress=None):
        asked.append(threading.current_thread() is threading.main_thread())
        clips = view.clips(node)
        return clips.with_typicality(np.arange(len(clips))[::-1].astype(float))

    monkeypatch.setattr(NodeView, "ranks_on_demand", property(lambda self: True))
    monkeypatch.setattr(type(handle), "clips", ranked)
    node = view.children(view.root)[0]
    n = len(view.clips(node))
    interpret.select(node)
    # The root's montage is showing until the ranked clips land, so wait for the node's count.
    assert _drain(qapp, lambda: bool(asked) and len(interpret.montage_order()) == n)
    assert asked == [False], "ranked on the UI thread"
    assert interpret.montage_order() == list(range(n))[::-1]


def test_the_label_bar_gives_the_clusters_figures(qapp, fitted):
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    node = next(n for n in view.children(view.root) if view.children(n))
    _select(qapp, interpret, node)
    text = interpret.label_bar().figures.text()
    n = len(view.clips(node))
    assert text.startswith(f"{n:,} clip{'s' if n != 1 else ''} · ")
    assert "% of the run" in text and "separation" not in text
    # The separation is the figures' tooltip's (2026-11-09): on row 1 it was what got cut.
    assert "run median" in interpret.label_bar().figures.toolTip()
    # The concentration is in the figures' tooltip only, and only when it beats its floor.
    flagged = any(
        view.distribution(node, groups).flagged for groups in interpret._groups.values()
    )
    assert bool(interpret.concentration) == flagged
    if flagged:
        assert interpret.concentration in interpret.label_bar().figures.toolTip()
    assert (
        "concentrated"
        not in " ".join(
            w.text() for w in interpret.label_bar().findChildren(QLabel)
        ).lower()
    ), "no chip on row 1"


def test_a_filter_shades_the_tree_and_greys_clips_by_the_apis_one_rule(qapp, fitted):
    """A selection never hides anything: every block is shaded by its matching share (the API's
    ``match_shares``), every clip still plays, drawn grey when it misses (``clip_matches``), and
    the hover's ✓ / ✗ per condition ANDs to exactly that."""
    from detail.api.cluster_filter import Selection, clip_matches

    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    assert interpret._icicle.shades() is None
    node = _most_clips(view)
    _select(qapp, interpret, node)
    left = frozenset((bx, by) for bx in range(14) for by in range(16))
    selection = Selection(cells=left)
    interpret.filters().set_selection(selection)
    shares = view.match_shares(selection)
    shaded = interpret._icicle.shades()
    assert shaded is not None and len(shaded) == view.tree.n_nodes
    for each, share in shaded.items():
        assert share == pytest.approx(shares[each], nan_ok=True)
    assert not interpret.filters().dock.key.ramp.isHidden()

    clips = view.clips(node)
    matching = clip_matches(selection, clips)
    assert 0 < matching.sum() < len(clips), "the fixture must split on the region"
    order = interpret.montage_order()
    assert [c.matching for c in interpret.montage()._clips] == [
        bool(matching[i]) for i in order
    ]
    for index, i in enumerate(order):
        hover = interpret._hover_text(index)
        assert ("✓ location" in hover) == bool(matching[i]), hover
        assert ("✗ location" in hover) != bool(matching[i]), hover
    # The count is of the run's finest clips: every leaf's, not the root's few long stretches.
    # (The fixture's leaves are shorter than its minimum instance, so it falls back to the root.)
    leaves = [view.clips(leaf) for leaf in range(view.tree.n_leaves)]
    leaves = [t for t in leaves if len(t)] or [view.clips(view.root)]
    total = sum(len(t) for t in leaves)
    matched = sum(int(clip_matches(selection, t).sum()) for t in leaves)
    assert interpret.filters().count_label.text() == (
        f"{matched:,} of {total:,} clips match"
    )

    interpret.filters().set_selection(Selection())
    assert interpret._icicle.shades() is None
    assert all(c.matching for c in interpret.montage()._clips)


def test_time_in_session_shades_by_time_inside_and_frames_clips_by_midpoint(
    qapp, fitted
):
    """Through the real page: the session spans come from each recording's timestamp map (read
    with the page), the tree is shaded by the API's trimmed shares, the montage's match frames
    are the midpoint rule, the count line and the section's own line count the same clips, and
    it ANDs with a region."""
    from detail.api import cluster_filter
    from detail.api.cluster_filter import Selection, clip_matches

    experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    spans = cluster_filter.session_spans(experiment.dir, view.recording_ids)
    assert spans and interpret._spans == spans, "the page read every recording's span"
    node = _most_clips(view)
    _select(qapp, interpret, node)
    clips = view.clips(node)
    # A range whose ends fall inside clips, so trimming and the midpoint both decide something.
    first, last = spans[view.recording_ids[0]]
    mids = ((clips.start_ns + clips.end_ns) // 2 - first) / (last - first)
    lo, hi = float(np.quantile(mids, 0.3)), float(np.quantile(mids, 0.7))
    lo, hi = round(lo, 2), round(hi, 2)
    selection = Selection(session=(lo, hi))
    interpret.filters().set_selection(selection)

    shares = view.match_shares(selection, spans=spans)
    shaded = interpret._icicle.shades()
    for each, share in shaded.items():
        assert share == pytest.approx(shares[each], nan_ok=True)
    assert 0 < shares[view.root] < 1
    matching = clip_matches(selection, clips, spans=spans)
    assert 0 < matching.sum() < len(clips), "the fixture must split on the range"
    order = interpret.montage_order()
    assert [c.matching for c in interpret.montage()._clips] == [
        bool(matching[i]) for i in order
    ]
    hover = interpret._hover_text(0)
    assert ("✓ time in session" in hover) == bool(matching[order[0]]), hover
    root = interpret._root_clips
    in_range = int(clip_matches(selection, root, spans=spans).sum())
    assert interpret.filters().session.span_label.text() == (
        f"{100 * lo:.0f}%–{100 * hi:.0f}% of each recording · "
        f"{in_range:,} clip{'s' if in_range != 1 else ''}"
    )
    assert interpret._split.pane_name(0) == "Filters · 1 active"

    left = frozenset((bx, by) for bx in range(14) for by in range(16))
    both = Selection(session=(lo, hi), cells=left)
    interpret.filters().set_selection(both)
    assert np.allclose(
        [interpret._icicle.shades()[n] for n in range(view.tree.n_nodes)],
        view.match_shares(both, spans=spans),
        equal_nan=True,
    )
    together = clip_matches(both, clips, spans=spans)
    assert np.array_equal(
        together, matching & clip_matches(Selection(cells=left), clips)
    )
    assert interpret._split.pane_name(0) == "Filters · 2 active"


def test_the_filters_dock_beside_the_tree_and_fold_away_remembered(qapp, fitted):
    """The filters are a column left of the tree, never a pop-up; the grip folds it to a strip
    reading "Filters · n active", and both its width and whether it is shut outlive the page.
    """
    from PyQt6.QtWidgets import QSplitter

    from detail.api.cluster_filter import Selection

    from detail.gui.cluster_filters import FilterDock
    from detail.gui.collapse import CollapsibleBand

    _experiment, _handle, workspace = _workspace(qapp, fitted)
    workspace.resize(1600, 1000)
    workspace.show()
    qapp.processEvents()
    interpret = workspace._interpret
    split = interpret._split
    assert isinstance(split, QSplitter)
    assert split.orientation() == Qt.Orientation.Horizontal
    assert isinstance(split.widget(0), FilterDock)
    assert split.widget(1).isAncestorOf(interpret._tree_handle)
    assert split.widget(1).findChild(CollapsibleBand) is None, "the tree does not fold"
    assert split.foldable_panes() == {0: "Filters"}, "the dock folds, the tree does not"
    assert not split.is_folded(0)

    split.setSizes([500, 1000])
    split.splitterMoved.emit(500, 1)  # what a drag of the handle does
    interpret.filters().set_selection(
        Selection(recording_ids=frozenset({"recording_001"}))
    )
    qapp.processEvents()
    assert split.pane_name(0) == "Filters · 1 active", "the strip counts what is on"
    split.toggle(0)  # what a double-click on the grip does
    qapp.processEvents()
    assert split.is_folded(0)
    assert split.sizes()[0] == 0 and split.handle(1).width() == split.strip_width()
    workspace.close()

    _experiment, _handle, again = _workspace(qapp, fitted)
    again.resize(1600, 1000)
    again.show()
    qapp.processEvents()
    split = again._interpret._split
    assert split.is_folded(0), "a fresh page opens shut, as the dock was left"
    split.toggle(0)
    qapp.processEvents()
    assert not split.is_folded(0)
    assert abs(split.sizes()[0] - 500) <= 2, "and with its width"
    again.close()


def test_double_clicking_a_tile_opens_the_clip_window_in_the_montages_order(
    qapp, fitted, monkeypatch
):
    from detail.gui import clip_popup

    opened = []

    def _open(opener, experiment, clip, *, sequence=None, index=0, **kw):
        opened.append((clip, sequence, index, kw.get("order_note")))

    monkeypatch.setattr(clip_popup, "open_clip_window", _open)
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    node = _most_clips(view)
    _select(qapp, interpret, node)
    interpret.montage().clip_activated.emit(2)
    clip, sequence, index, note = opened[0]
    clips = view.clips(node)
    i = interpret.montage_order()[2]
    assert index == 2 and sequence[2] is clip
    assert (clip.start_t_ns, clip.end_t_ns) == (
        int(clips.start_ns[i]),
        int(clips.end_ns[i]),
    )
    assert len(sequence) == len(clips) and note == "most typical first"


def test_the_clip_list_export_writes_every_clip_in_the_montages_order(
    qapp, fitted, tmp_path
):
    """**Export ▾**'s CSV list, written for real: every clip of the cluster, in the montage's
    order, ranked by place (``montage_export``)."""
    import csv

    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    _select(qapp, interpret, _most_clips(view))
    exports = interpret.montage().exports
    job = exports.job()
    clips = interpret.export_clips()
    assert job.plain == clips
    assert len(clips) == len(view.clips(interpret._selected))
    assert [c.rank for c in clips] == list(range(1, len(clips) + 1))
    path = tmp_path / "clips.csv"
    exports.write_list(job, path)
    assert _drain(qapp, lambda: path.exists() and not exports.written.isHidden())
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(clips)


def test_the_menu_has_no_clip_windows_of_its_own(qapp, fitted):
    """The montage under the tree plays every clip; the tree's menu opens no second window."""
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    menu = interpret._icicle.context_menu(interpret._view.root)
    texts = [a.text() for a in menu.actions()]
    assert not any("Preview" in t or "montage" in t.lower() for t in texts)


# --- missing previews block the montage, and only the montage --------------------------------


class _Queue:
    """The preview queue's surface the page uses, with signals a test fires."""

    def __init__(self, experiment):
        from PyQt6.QtCore import QObject, pyqtSignal

        class _Signals(QObject):
            progress = pyqtSignal(int, str)
            recording_done = pyqtSignal(str)
            failed = pyqtSignal(str, object)

        self._signals = _Signals()
        self.progress = self._signals.progress
        self.recording_done = self._signals.recording_done
        self.failed = self._signals.failed
        self.experiment = experiment
        self.current = None
        self.permille = 0
        self.status = ""
        self.phase = ""
        self.asked: list = []
        self.withdrawn: list = []

    def prioritise(self, ids):
        self.asked.append(list(ids))

    def withdraw(self, ids):
        self.withdrawn.append(list(ids))


def _missing_previews(monkeypatch, experiment):
    from detail.api import clip_tiles
    from detail.gui import preview_jobs

    queue = _Queue(experiment)
    monkeypatch.setattr(preview_jobs, "current", lambda: queue)
    monkeypatch.setattr(
        clip_tiles,
        "missing",
        lambda _e, _t, recording_ids: {"proxy": list(recording_ids), "masks": []},
    )
    return queue


def test_missing_previews_block_the_montage_with_progress_and_the_tree_stays_usable(
    qapp, fitted, monkeypatch
):
    experiment, _run_id = fitted
    queue = _missing_previews(monkeypatch, experiment)
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    montage = interpret.montage()
    ids = list(view.recording_ids)
    assert queue.asked == [ids], "this run's recordings go to the front of the queue"
    assert montage.is_blocked()
    assert f"0 of {len(ids)} recordings" in montage._overlay._label.text()
    queue.current, queue.permille, queue.phase = ids[0], 500, "proxy · part 1 of 1"
    queue.progress.emit(500, "proxy")
    assert "proxy · part 1 of 1" in montage._overlay._label.text()

    # The tree, the labels and the filters do not wait.
    node = view.children(view.root)[0]
    _select(qapp, interpret, node)
    assert interpret._selected == node and montage.is_blocked()
    assert interpret._label_field.edit.isEnabled()

    for rid in ids:
        queue.recording_done.emit(rid)
    assert not montage.is_blocked()
    assert montage.source() is not None


def test_cancel_stops_waiting_for_this_runs_previews_only(qapp, fitted, monkeypatch):
    experiment, _run_id = fitted
    queue = _missing_previews(monkeypatch, experiment)
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    ids = list(interpret._view.recording_ids)
    interpret._cancel_previews()
    assert queue.withdrawn == [ids], "only this run's recordings are withdrawn"
    assert not interpret.montage().is_blocked()
    assert not interpret._preview_note.isHidden()
    interpret._build_previews.click()
    assert queue.asked[-1] == ids and interpret.montage().is_blocked()


# --- leaving the page and coming back mid-build ------------------------------------------------
#
# The class: a page that re-reads what it waits for from disk while a build is under way sees
# the build in hand half-landed. A recording whose proxy is written and whose masks are still
# building reads as "missing masks only", so a page reopened at that moment counted only the
# *other* recording, called this run's own build "the build in hand" ("0 of 1 · waiting for the
# build in hand to finish") and drew an empty bar beside the status bar's 60% for the same build.
# These drive the real shell: the page is left through the path line and through another page
# (which destroys it), and opened again through its own card.


def _shell(qapp, fitted, monkeypatch, tmp_path):
    from detail.gui.app import MainWindow
    from detail.storage import locations

    root = tmp_path / "experiments_root"
    root.mkdir()
    monkeypatch.setattr(locations, "experiments_root", lambda: root)
    window = MainWindow()
    window.resize(1500, 950)
    window.show()
    return window


def _open_interpret(qapp, window, fitted):
    from detail.gui.pages import AnalysisRunDetailPage

    experiment, run_id = fitted
    page = AnalysisRunDetailPage(experiment, run_id)
    window._show(page, run_id)
    page._cards["Interpret"].click()
    interpret = page._result._interpret
    assert _drain(qapp, lambda: interpret._view is not None)
    return page, interpret


def _leave_and_return(qapp, window, fitted):
    """Back to the run's landing through the path line, into INTERPRET again, then away to
    another page (which destroys this one) and back through the run's card."""
    from PyQt6.QtWidgets import QLabel

    window._context.linkActivated.emit("landing")
    page = window._content.currentWidget()
    page._cards["Interpret"].click()
    window._show(QLabel("elsewhere"), "elsewhere")
    assert _drain(qapp, lambda: window._content.count() == 1)
    return _open_interpret(qapp, window, fitted)


def _blocked_text(interpret) -> str:
    overlay = interpret.montage()._overlay
    return overlay._label.text() if overlay is not None else ""


def test_returning_mid_build_names_this_runs_build_and_shows_its_progress(
    qapp, fitted, monkeypatch, tmp_path
):
    from detail.api import clip_tiles

    queue = _missing_previews(monkeypatch, fitted[0])
    window = _shell(qapp, fitted, monkeypatch, tmp_path)
    try:
        _page, interpret = _open_interpret(qapp, window, fitted)
        own = list(interpret._view.recording_ids)[0]
        # Another recording of the run waits behind this one; this one's proxy has landed and
        # its masks are building.
        monkeypatch.setattr(
            clip_tiles,
            "missing",
            lambda _e, _t, _ids: {"proxy": ["recording_later"], "masks": [own]},
        )
        phase = f"Building previews — masks track_02 · {own} · part 1 of 1 · 3 of 5"
        queue.current, queue.permille, queue.phase = own, 600, phase
        _page, interpret = _leave_and_return(qapp, window, fitted)
        queue.progress.emit(600, phase)
        text = _blocked_text(interpret)
        assert phase in text
        assert "waiting for the build in hand" not in text
        assert "0 of 2 recordings" in text
        # The aggregate fold: (done + the build in hand's fraction) / total.
        assert interpret._wait.fraction() == pytest.approx((0 + 0.6) / 2)
        bar = interpret.montage()._overlay._bar
        assert bar.value() > 0
        assert bar.value() / bar.maximum() == pytest.approx(0.3, abs=0.01)
        assert queue.withdrawn == [], "leaving cancelled this run's build"
        # It lands: one of two done, and the next one is waited for.
        queue.current, queue.permille = None, 0
        queue.recording_done.emit(own)
        assert "1 of 2 recordings" in _blocked_text(interpret)
    finally:
        window.close()


def test_returning_while_another_recording_builds_says_it_waits(
    qapp, fitted, monkeypatch, tmp_path
):
    queue = _missing_previews(monkeypatch, fitted[0])
    window = _shell(qapp, fitted, monkeypatch, tmp_path)
    try:
        queue.current, queue.permille, queue.phase = "not_this_run", 600, "proxy"
        _page, interpret = _open_interpret(qapp, window, fitted)
        _page, interpret = _leave_and_return(qapp, window, fitted)
        queue.progress.emit(600, "proxy")
        text = _blocked_text(interpret)
        total = len(interpret._view.recording_ids)
        assert f"0 of {total} recordings" in text
        assert "waiting for the build in hand to finish" in text
        assert interpret._wait.fraction() == 0.0
    finally:
        window.close()


def test_leaving_and_returning_leaves_one_listener(qapp, fitted, monkeypatch, tmp_path):
    """Five leave/return cycles: each of the queue's signals reaches exactly one page — the one
    on screen — and the line it draws is the one the first return drew. A page's connection
    outlives its page only until that signal next fires (PyQt drops a dead receiver on
    delivery and frees it at the next event-loop pass), so the listener count is read after a
    round of every signal and a pump."""
    from detail.api import clip_tiles
    from detail.gui import node_workspace

    queue = _missing_previews(monkeypatch, fitted[0])
    signals = queue._signals
    window = _shell(qapp, fitted, monkeypatch, tmp_path)
    reached: list[str] = []
    for name in (
        "_on_preview_progress",
        "_on_preview_done",
        "_on_preview_failed",
        "_on_masks_done",
    ):
        real = getattr(node_workspace.NodeInterpretTab, name)

        def counted(self, *args, _real=real, _name=name):
            reached.append(_name)
            return _real(self, *args)

        monkeypatch.setattr(node_workspace.NodeInterpretTab, name, counted)

    def one_round():
        reached.clear()
        queue.progress.emit(queue.permille, queue.phase)
        queue.recording_done.emit("not_this_run")
        queue.failed.emit("not_this_run", "why")
        return sorted(reached)

    def listeners():
        one_round()
        _drain(qapp, lambda: False, 100)
        return (
            signals.receivers(queue.progress),
            signals.receivers(queue.recording_done),
            signals.receivers(queue.failed),
        )

    try:
        _page, interpret = _open_interpret(qapp, window, fitted)
        own = list(interpret._view.recording_ids)[0]
        monkeypatch.setattr(
            clip_tiles,
            "missing",
            lambda _e, _t, _ids: {"proxy": ["recording_later"], "masks": [own]},
        )
        queue.current, queue.permille, queue.phase = own, 250, "masks"
        _page, interpret = _leave_and_return(qapp, window, fitted)
        first = (one_round(), listeners(), _blocked_text(interpret))
        assert first[0] == [
            "_on_masks_done",
            "_on_preview_done",
            "_on_preview_failed",
            "_on_preview_progress",
        ]
        # One page: its wait on progress / done / failed, and its masks on done.
        assert first[1] == (1, 2, 1), f"listeners {first[1]}"
        for cycle in range(5):
            _page, interpret = _leave_and_return(qapp, window, fitted)
            now = (one_round(), listeners(), _blocked_text(interpret))
            assert now == first, f"cycle {cycle + 1}: {now} != {first}"
        assert queue.withdrawn == [], "leaving cancelled this run's build"
    finally:
        window.close()


# --- the label field --------------------------------------------------------------------------


def _menu_entries(field):
    return {
        action.text(): action.menu()
        for action in field.labels_menu().actions()
        if action.menu() is not None
    }


def test_the_label_list_shows_every_label_with_its_time_and_labels_from_it(
    qapp, fitted
):
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    a, b = view.children(view.root)
    _label(qapp, interpret, a, "moving", handle)
    interpret.select(b)
    entries = _menu_entries(interpret._label_field)
    (text,) = [t for t in entries if t.startswith("moving · ")]
    assert text.endswith("%")
    actions = {act.text(): act for act in entries[text].actions()}
    actions["Label as “moving”"].trigger()
    field = interpret._label_field
    assert field.edit.text() == "moving" and str(b) not in interpret._names
    field.label_button.click()
    assert interpret._names[str(b)] == "moving"
    _settle(qapp, interpret)
    assert interpret._key_row.swatches.texts()[0].startswith("■ moving")


def test_a_label_the_cluster_cannot_join_is_greyed_with_the_reason(qapp, fitted):
    """A label's clusters share one parent label. A label living inside "moving" is listed for a
    cluster outside it greyed, with why — not offered and then refused."""
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    a, b = view.children(view.root)
    inner = view.children(a)[0]
    _label(qapp, interpret, a, "moving", handle)
    _label(qapp, interpret, inner, "walking", handle)
    interpret.select(b)
    entries = _menu_entries(interpret._label_field)
    walking = next(menu for text, menu in entries.items() if "walking" in text)
    join = walking.actions()[0]
    assert not join.isEnabled() and "share one parent label" in join.toolTip()
    moving = next(m for t, m in entries.items() if t.startswith("moving"))
    assert moving.actions()[0].isEnabled()


def test_rename_and_merge_are_dead_when_the_run_cannot_be_edited(qapp, fitted):
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    _label(qapp, interpret, view.children(view.root)[0], "moving", handle)
    interpret._can_edit = False
    interpret._label_field.set_editable(False)
    menu = next(iter(_menu_entries(interpret._label_field).values()))
    texts = {a.text(): a.isEnabled() for a in menu.actions() if a.text()}
    assert texts["Rename “moving”…"] is False
    assert texts["Merge into another “moving”…"] is False
    interpret._rename_label(
        "anything"
    )  # a guard, not a dialog: nothing opens, nothing fails


def test_the_tree_height_is_kept_for_next_time(qapp, fitted):
    """A drag wins over fitting every level, and is kept; from then on rows keep their full
    height and the tree scrolls."""
    from detail.gui.node_workspace import TREE_HEIGHT_KEY

    experiment, run_id = fitted
    _experiment, _handle, first = _workspace(qapp, fitted)
    interpret = first._interpret
    assert interpret._icicle.fits_rows(), "until a drag, rows fit"
    interpret._resize_tree(37)
    height = interpret._tree_holder.maximumHeight()
    assert not interpret._icicle.fits_rows()
    assert int(QSettings().value(TREE_HEIGHT_KEY)) == height
    first.deleteLater()
    qapp.processEvents()
    second = ClusterWorkspace(experiment, experiment.authored_run(run_id))
    assert _drain(qapp, lambda: second._interpret._view is not None)
    assert second._interpret._tree_holder.maximumHeight() == height
    assert not second._interpret._icicle.fits_rows()


def _shown_workspace(qapp, fitted, height: int):
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    workspace.resize(1400, height)
    workspace.show()
    interpret = workspace._interpret
    _drain(qapp, lambda: False, 0.3)
    return workspace, interpret


@pytest.mark.parametrize("height", [1000, 700], ids=["room", "tight"])
def test_the_tree_opens_showing_every_level_and_the_montage_keeps_two_rows(
    qapp, fitted, height
):
    """The user saw two rows of a twelve-level tree: it opened at a remembered height. It opens
    at the height for every level now, and when that would leave the montage under two rows of
    tiles, the rows shrink instead of scrolling. A height remembered by an earlier build (the
    old key) is ignored."""
    from detail.gui import node_icicle
    from detail.gui.node_workspace import MONTAGE_MIN_ROWS

    QSettings().setValue("interpret/tree_height", 60)
    workspace, interpret = _shown_workspace(qapp, fitted, height)
    try:
        icicle, montage = interpret._icicle, interpret._montage
        levels = icicle.levels()
        assert levels >= 3, "the fixture's tree must have levels to cut off"
        assert icicle.scroll_range() == 0, "every level is on screen"
        assert icicle.content_height() <= icicle.height()
        cell = montage._cell().height()
        assert montage.grid.viewport().height() >= MONTAGE_MIN_ROWS * cell
        if icicle._row_height() < node_icicle.row_px():
            # Shrunk only because the montage needed the room.
            assert interpret._tree_holder.height() < (
                interpret._tree_holder.height()
                - icicle.height()
                + levels * node_icicle.row_px()
            )
    finally:
        workspace.close()
        workspace.deleteLater()


def test_rows_shrink_rather_than_scroll_when_the_window_is_short(qapp, fitted):
    """A tree pane shorter than every level at full row height shrinks its rows (down to
    ``min_row_px``) instead of scrolling. The pane is squeezed directly: the page's own minimum
    height, not the window, decides how tall the workspace is, so a short window alone does not
    make the fixture's ten-level tree tight."""
    from detail.gui import node_icicle

    workspace, interpret = _shown_workspace(qapp, fitted, 560)
    try:
        icicle = interpret._icicle
        levels = icicle.levels()
        icicle.resize(icicle.width(), int(levels * node_icicle.row_px() * 0.7))
        assert icicle._row_height() < node_icicle.row_px(), "the pane must be tight"
        assert icicle._row_height() >= node_icicle.min_row_px()
        if icicle._row_height() > node_icicle.min_row_px():
            assert icicle.scroll_range() == 0
    finally:
        workspace.close()
        workspace.deleteLater()


def test_there_is_no_ethogram_on_interpret(qapp, fitted):
    from detail.gui.cluster_strip import EthogramStrip

    _experiment, _handle, workspace = _workspace(qapp, fitted)
    assert not workspace._interpret.findChildren(EthogramStrip)


# --- BROWSE carries the selected cluster's lane -----------------------------------------------


def test_the_selected_cluster_is_a_lane_under_browses_ethogram(qapp, fitted):
    """10/4: click a cluster and see where it falls in each video, labelled or not — on BROWSE
    now, where the ethogram is. The lane is exactly the node's stretches and follows the
    selection; the root clears it."""
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret, browse = workspace._interpret, workspace._browse
    view = interpret._view
    first, second = view.children(view.root)[:2]
    _label(qapp, interpret, first, "moving", handle)
    assert _drain(qapp, lambda: browse._loaded and bool(browse._drawn))

    def expected(node):
        return sorted(
            (r["recording_id"], r["start_ns"], r["end_ns"])
            for r in view.labelled_runs({str(node): "x"})
        )

    def shown():
        return sorted(
            (r["recording_id"], r["start_ns"], r["end_ns"])
            for rows in browse._lane_rows
            for r in rows
        )

    # BROWSE reads the lane once it is shown, off the UI thread (a node click walks every
    # leaf run), so a click while it is hidden reads nothing until the tab is opened.
    workspace.resize(1400, 900)
    workspace.show()
    try:
        workspace._tabs.setCurrentWidget(browse)
        interpret.select(second)
        assert _drain(qapp, lambda: shown() == expected(second)) and shown()
        assert sum(len(m) for m in browse._ethogram._lane) == len(shown())
        workspace._tabs.setCurrentWidget(interpret)
        interpret.select(first)
        _drain(qapp, lambda: False, 300)
        assert shown() == expected(second), "read while BROWSE was hidden"
        workspace._tabs.setCurrentWidget(browse)
        assert _drain(qapp, lambda: shown() == expected(first))
        interpret.select(view.root)
        assert browse._ethogram._lane is None
    finally:
        workspace.hide()


def test_missing_masks_alone_do_not_block_the_footage(qapp, fitted, monkeypatch):
    """With the proxy built and only the masks to come, the montage plays footage at once and
    its source is remade (outlines on) when the recording's masks land."""
    from detail.api import clip_tiles
    from detail.gui import preview_jobs

    experiment, _run_id = fitted
    queue = _Queue(experiment)
    monkeypatch.setattr(preview_jobs, "current", lambda: queue)
    monkeypatch.setattr(
        clip_tiles,
        "missing",
        lambda _e, _t, recording_ids: {"proxy": [], "masks": list(recording_ids)},
    )
    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    ids = list(interpret._view.recording_ids)
    montage = interpret.montage()
    assert not montage.is_blocked() and queue.asked == [ids]
    assert "outline" in interpret._preview_text.text()
    before = montage.source()
    for rid in ids:
        queue.recording_done.emit(rid)
    assert montage.source() is not before
    assert interpret._preview_note.isHidden()


# --- clips removed on AUDIT, shown on INTERPRET ------------------------------------------------


def _temporal_workspace(qapp, fitted_temporal):
    experiment, run_id = fitted_temporal
    handle = experiment.temporal_clustering_run(run_id)
    view = handle.node_view()
    first, second = view.children(view.root)[:2]
    handle.name_nodes({first: "climb", second: "rear"})
    rows = view.clips(first).rows()
    handle.take_out_clips(rows[:2], "climb")
    handle.take_out_clips(rows[2:3], "climb", "rear")
    _e, _h, workspace = _workspace(qapp, fitted_temporal, handle=handle)
    return handle, workspace, first


def test_a_removed_clip_is_faded_and_tagged_in_place_on_interpret(
    qapp, fitted_temporal
):
    handle, workspace, first = _temporal_workspace(qapp, fitted_temporal)
    interpret = workspace._interpret
    interpret.select(first)
    assert _drain(qapp, lambda: len(interpret.montage().clips()) > 0)
    clips = interpret.montage().clips()
    n = len(handle.node_view().clips(first))
    assert len(clips) == n, "a removed clip stays in its cluster"
    tags = sorted(c.tag for c in clips if c.faded)
    assert tags == ["not climb", "not climb", "→ rear"]
    assert sorted(c.crossed for c in clips if c.faded) == [False, True, True]
    assert all("Right-click to put it back" in c.note for c in clips if c.faded)
    assert f"{n:,} clips · 3 removed" in interpret.label_bar().figures.text()


def test_interpret_rings_and_reads_a_cluster_only_once_every_clip_is_reviewed(
    qapp, fitted_temporal
):
    """AUDIT's census reaches INTERPRET: before ``finish_review`` no ring and no AUDIT line; after
    it, the ring and the exact share."""
    experiment, run_id = fitted_temporal
    handle = experiment.temporal_clustering_run(run_id)
    view = handle.node_view()
    first, second = view.children(view.root)[:2]
    handle.name_nodes({first: "climb", second: "rear"})
    rows = view.clips(first).rows()
    handle.take_out_clips(rows[:1], "climb")
    _e, _h, workspace = _workspace(qapp, fitted_temporal, handle=handle)
    interpret = workspace._interpret
    assert interpret._icicle.outlines() == {}
    interpret.select(first)
    assert interpret._audit_block.isHidden()
    handle.finish_review(first)
    interpret.refresh_audit()
    assert _drain(qapp, lambda: set(interpret._icicle.outlines()) == {first})
    interpret.select(first)
    text = interpret._audit_block._figure.text()
    n = len(rows)
    assert text == f"precision {(n - 1) / n:.2f} · all {n} clips reviewed"


def test_right_click_put_back_reaches_the_run(qapp, fitted_temporal, monkeypatch):
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QMenu

    handle, workspace, first = _temporal_workspace(qapp, fitted_temporal)
    interpret = workspace._interpret
    interpret.select(first)
    montage = interpret.montage()
    assert _drain(qapp, lambda: any(c.faded for c in montage.clips()))
    index = next(i for i, c in enumerate(montage.clips()) if c.tag == "→ rear")
    menus = []
    with monkeypatch.context() as patch:
        patch.setattr(QMenu, "popup", lambda self, *_a: menus.append(self))
        montage.clip_context_requested.emit(index, QPoint(0, 0))
    (menu,) = menus
    (action,) = menu.actions()
    assert action.text() == "Put back in climb" and action.isEnabled()
    action.trigger()
    assert not montage.clips()[index].faded, "the fade lifts at once"
    assert _drain(qapp, lambda: len(handle.taken_out("climb")) == 2)
    assert _drain(qapp, lambda: not interpret.writes_pending())


def _two_with_clips(view) -> tuple[int, int]:
    """Two non-root clusters with clips, the busier first."""
    nodes = sorted(
        (n for n in range(view.tree.n_nodes) if n != view.root and len(view.clips(n))),
        key=lambda n: -len(view.clips(n)),
    )
    assert len(nodes) >= 2, "the fit has two clusters with clips"
    return nodes[0], nodes[1]


def test_a_filter_set_while_a_cluster_ranks_waits_for_its_clips(qapp, fitted):
    """The reported sequence: a cluster's clips are showing, another is selected and starts
    ranking off the UI thread (the montage is emptied), and a filter is set before the ranking
    lands. The filter used to be applied to the *previous* cluster's clips, which the page still
    held, against a montage holding none — ``ValueError: 21 flags for 0 clips`` in a slot. The
    page's clips are emptied with the montage, and the filter reaches the new clips when they
    land."""
    from detail.api.cluster_filter import Selection, clip_matches

    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    assert view.ranks_on_demand, "the window under test only exists while clips rank"
    first, second = _two_with_clips(view)
    _select(qapp, interpret, first)
    assert len(interpret.montage().clips()) == len(view.clips(first))

    interpret.select(second)
    assert interpret._clip_text.startswith("ranking"), "the ranking has not landed yet"
    assert interpret.montage().clips() == []
    assert interpret.montage_order() == [], "the page's order agrees with the montage"
    selection = Selection(session=(0.25, 0.75))
    interpret.filters().set_selection(selection)
    assert interpret.montage().clips() == [], "nothing shown before the ranking lands"

    assert _drain(qapp, lambda: not interpret._clip_text.startswith("ranking"))
    clips = handle.clips(second)
    matching = clip_matches(selection, clips, spans=interpret._spans)
    assert 0 < matching.sum() < len(clips), "the fixture must split on the range"
    order = interpret.montage_order()
    assert len(order) == len(clips)
    assert [c.matching for c in interpret.montage().clips()] == [
        bool(matching[i]) for i in order
    ]


def test_an_order_picked_while_a_cluster_ranks_does_not_bring_back_the_last_ones(
    qapp, fitted
):
    """The same window, the order and context controls: either one re-showed the page's clips,
    which while a cluster ranked were still the previous cluster's — its tiles came back under
    "ranking…" until the new ones replaced them."""
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    first, second = _two_with_clips(view)
    _select(qapp, interpret, first)

    interpret.select(second)
    assert interpret._clip_text.startswith("ranking"), "the ranking has not landed yet"
    montage = interpret.montage()
    other = next(k for k in montage.order_actions if k != montage.order())
    montage.order_actions[other].trigger()
    assert montage.clips() == [], "the previous cluster's tiles came back"
    interpret._on_context()
    assert montage.clips() == [], "the previous cluster's tiles came back"

    assert _drain(qapp, lambda: not interpret._clip_text.startswith("ranking"))
    assert (
        interpret.montage_order()
        == handle.clips(second).order(other, seed=second).tolist()
    )


# --------------------------------------------------------------------------- #
# the Differences line on the page: a saved screen's kept clusters shade the tree (§22)
# --------------------------------------------------------------------------- #


class _FakeScreen:
    """A saved node screen as the page reads it: an id, a record, a ranking whose kept items
    are ``kept`` and whose ``clearing`` count clears the floor (the rest nested, on nodes the
    tree does not have). ``rankings`` counts the re-rankings: seconds each over a real family,
    so the page ranks a screen once per read."""

    def __init__(
        self, comparison_id: str, kept: list[int], clearing: int, leads=()
    ) -> None:
        self.id = comparison_id
        self._kept = list(kept)
        self._clearing = max(clearing, len(kept))
        #: Clusters inside the floor at 5% that clear at p ≤ 0.2: what a looser reading
        #: shows (§22).
        self._leads = list(leads)
        self.rankings = 0

    def record(self) -> dict:
        return {
            "source": "nodes",
            "source_run_id": "an_01",
            "design": "paired",
            "design_column": "drug",
            "levels": ["vehicle", "drug"],
        }

    def ranking(self):
        from detail.analysis import compare

        self.rankings += 1
        nested = [100_000 + i for i in range(self._clearing - len(self._kept))]

        def item(n: int, effect: float, family_p: float, own_p: float, q: float):
            return compare.Effect(
                measure_key=("time_fraction", f"node:{n}", "", "whole"),
                effect=effect,
                n_units=10,
                n_present=10,
                consistency=10,
                loo_sign_stable=True,
                family_p=family_p,
                own_p=own_p,
                q=q,
            )

        items = [
            *(item(n, math.log(4.0), 0.02, 0.004, 0.5) for n in self._kept),
            *(item(n, math.log(4.0), 0.03, 0.004, 0.5) for n in nested),
            *(item(n, math.log(2.5), 0.2, 0.03, 0.6) for n in self._leads),
        ]
        # The nested ones sit under the first kept cluster, so a reading at any level
        # peak-picks them away as the saved screen did (`suppress_nesting`).
        ancestors = {
            f"node:{n}": {f"node:{self._kept[0]}"} if self._kept else set()
            for n in nested
        }
        # The null's 95th percentile (index 14 of 16) is the floor, 1.0: the kept and nested
        # items (ln 4) clear it, a lead (ln 2.5) does not — as its family-wise p says.
        return compare.Ranking(
            contrasts=items,
            floor=1.0,
            draws=16,
            seed=0,
            floor_distribution=[0.05 * i for i in range(14)] + [1.0, 1.5],
            design=compare.PAIRED,
            levels=("vehicle", "drug"),
            unit=("ID",),
            n_units=10,
            min_units=5,
            exact=True,
            kept=tuple(items[i].measure_key for i in range(len(self._kept))),
            ancestors=ancestors,
        )

    def kept_nodes(self) -> list[int]:
        import detail.api.comparison as comparison_api

        return comparison_api.kept_nodes_of(self.ranking())


def test_without_nesting_suppression_every_clearing_cluster_is_kept_not_nested():
    """``Ranking.kept`` is ``None`` when no suppression was applied; the findings are then
    every clearing item (``kept_items``), and a hover calling them "nested under a kept
    cluster" names a parent that does not exist."""
    from detail.analysis import compare
    from detail.gui.node_workspace import _node_effects

    screen = _FakeScreen("cmp_08", [3, 5], clearing=3)
    ranking = screen.ranking()
    suppressed = _node_effects(ranking, screen.record())
    assert {node: row.kept for node, row in suppressed.items()} == {
        3: True,
        5: True,
        100_000: False,
    }, "with suppression, a clearing item outside the kept set is nested"
    unsuppressed = compare.Ranking(
        contrasts=ranking.contrasts,
        floor=ranking.floor,
        draws=ranking.draws,
        seed=ranking.seed,
        design=ranking.design,
        levels=ranking.levels,
        unit=ranking.unit,
        n_units=ranking.n_units,
        min_units=ranking.min_units,
        kept=None,
    )
    effects = _node_effects(unsuppressed, screen.record())
    assert {node: row.kept for node, row in effects.items()} == {
        3: True,
        5: True,
        100_000: True,
    }


def _screens_read(qapp, interpret) -> None:
    from tests.gui.conftest import drain_until

    assert drain_until(qapp, lambda: interpret._screens_reader is None, 20_000)


def _kept_candidate(view) -> int:
    """A cluster under the root with clips of its own — what a screen would keep. The
    fixture's busiest cluster is the root itself (one recording, one stretch)."""
    for node in range(view.tree.n_nodes - 1):
        if len(view.clips(node)) and view.summary(node).parent >= 0:
            return node
    raise AssertionError("the fixture must have a cluster under the root with clips")


def test_a_saved_screens_kept_clusters_shade_the_tree_through_the_filters(
    qapp, fitted, monkeypatch
):
    """Ticking a screen in the Differences line is the clusters condition: a kept cluster
    shades 100%, its parent by the time under it, the rest grey, through the same
    ``match_shares`` and ``clip_matches`` every other filter goes through."""
    import detail.api.comparison as comparison_api
    from detail.api.cluster_filter import CLUSTERS, Selection

    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    _screens_read(qapp, interpret)
    node = _kept_candidate(view)
    screen = _FakeScreen("cmp_07", [node], clearing=3)
    monkeypatch.setattr(comparison_api, "node_screens", lambda exp, run_id: [screen])
    interpret._read_screens()
    _screens_read(qapp, interpret)

    bar = interpret.filters()
    lst = bar.screens.list
    assert lst.count() == 1
    assert lst.item(0).data(Qt.ItemDataRole.UserRole) == "cmp_07"
    assert lst.item(0).text() == "drug vs vehicle · paired   1 kept of 3 clearing"
    assert screen.rankings == 1, "a screen is ranked once per read, not once per use"

    item = lst.item(0)
    item.setCheckState(Qt.CheckState.Checked)
    selection = bar.selection()
    assert selection.keys() == [CLUSTERS] and selection.leaves_source == "cmp_07"
    assert selection.leaves == view.leaves_under_nodes([node])
    # Alone, the screen shades by its effect (the next test); beside a clip filter the
    # share ramp returns, with the screen's leaves still ANDed in.
    bar.set_selection(
        Selection(
            leaves=selection.leaves,
            leaves_source="cmp_07",
            recording_ids=frozenset(view.recording_ids),
        )
    )
    selection = bar.selection()
    shaded = interpret._icicle.shades()
    assert shaded[node] == pytest.approx(1.0)
    expected = view.match_shares(selection)
    for each, share in shaded.items():
        assert share == pytest.approx(expected[each], nan_ok=True)
    assert 0.0 < shaded[view.root] < 1.0
    assert bar.count_label.text().endswith("clips match")
    from detail.gui.cluster_filters import panel_name

    assert panel_name(selection) == "Filters · 2 active"

    bar.field_row("Differences").reset_button.click()
    assert bar.selection().leaves is None
    bar.field_row("Video").reset_button.click()
    assert interpret._icicle.shades() is None


def test_run_a_new_screen_opens_the_launch_dialog_seeded_and_ticks_what_it_saved(
    qapp, fitted, monkeypatch
):
    import detail.api.comparison as comparison_api
    import detail.gui.comparison as comparison_gui
    from PyQt6.QtWidgets import QDialog

    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    _screens_read(qapp, interpret)
    node = _kept_candidate(view)
    seeds = []

    class _Dialog:
        def __init__(self, experiment, parent=None, seed=None) -> None:
            seeds.append(seed)
            self.comparison = _FakeScreen("cmp_09", [node], clearing=1)

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(comparison_gui, "ComparisonLaunchDialog", _Dialog)
    monkeypatch.setattr(
        comparison_api,
        "node_screens",
        lambda exp, run_id: [_FakeScreen("cmp_09", [node], clearing=1)],
    )
    interpret.filters().screens.run_button.click()
    assert seeds == [{"source_run_id": str(interpret._run.id), "source": "nodes"}]
    _screens_read(qapp, interpret)
    assert interpret.filters().selection().leaves_source == "cmp_09"
    assert interpret.filters().selection().leaves == view.leaves_under_nodes([node])
    assert interpret._icicle._diverging, "ticked alone, the screen shades by its effect"


def test_a_screen_alone_shades_the_tree_by_its_effect_and_a_clip_filter_brings_the_share_back(
    qapp, fitted, monkeypatch
):
    """The effect mode (§22): while the ticked screen is the only filter on, the tree reads the
    screen's statistic on the diverging scale and the key its axis and floor; tick a clip filter
    too and both return to the share ramp, because two scales on one tree are unreadable.
    """
    import math

    import detail.api.comparison as comparison_api
    from detail.api.cluster_filter import Selection
    from detail.gui import plot_theme
    from detail.gui.match_key import LOG_RATIO_EXTENT

    _experiment, _handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    _screens_read(qapp, interpret)
    node = _kept_candidate(view)

    class _Screen(_FakeScreen):
        def ranking(self):
            from detail.analysis import compare

            key = ("time_fraction", f"node:{node}", "", "whole")
            item = compare.Effect(
                measure_key=key,
                effect=math.log(12.4),
                n_units=10,
                n_present=6,
                consistency=7,
                loo_sign_stable=True,
            )
            return compare.Ranking(
                contrasts=[item],
                floor=math.log(7.14),
                draws=1024,
                seed=0,
                design=compare.PAIRED,
                levels=("vehicle", "drug"),
                unit=("ID",),
                n_units=10,
                min_units=5,
                exact=True,
                kept=(key,),
            )

    monkeypatch.setattr(
        comparison_api,
        "node_screens",
        lambda exp, run_id: [_Screen("cmp_11", [node], 1)],
    )
    interpret._read_screens()
    _screens_read(qapp, interpret)
    bar = interpret.filters()
    bar.screens.list.item(0).setCheckState(Qt.CheckState.Checked)
    interpret.select(node)

    assert interpret._icicle._diverging
    shaded = interpret._icicle.shades()
    assert shaded[node] == pytest.approx(math.log(12.4) / LOG_RATIO_EXTENT)
    assert shaded[view.root] != shaded[view.root], "the root has no row: NaN, grey"
    key = bar.dock.key
    assert key.is_effect()
    assert key.text().startswith(
        "Less ↔ more under drug (against vehicle) | ÷20 … ×20 |"
    )
    assert "selected 12.4×" in key.text() and "floor" in key.text()
    assert "12.4×" in interpret._icicle._describe(node)
    assert "kept" in interpret._icicle._describe(node)
    theme = plot_theme.current()
    assert interpret._icicle.fill_of(node)[0] == plot_theme.effect_colour(
        shaded[node], theme
    )

    # A clip filter beside the screen: back to the share ramp, the screen still ANDed in.
    bar.set_selection(
        Selection(
            leaves=bar.selection().leaves,
            leaves_source="cmp_11",
            recording_ids=frozenset(view.recording_ids),
        )
    )
    assert not interpret._icicle._diverging
    assert not key.is_effect()
    assert interpret._icicle.shades()[node] == pytest.approx(1.0)
    assert key.text().startswith("Clips that match | 0% … 100% |")


# --- a label written while a screen is ticked remembers the screen (§22, Phase 3) ---------


def _tick_screen(qapp, interpret, monkeypatch, screen) -> None:
    import detail.api.comparison as comparison_api

    _screens_read(qapp, interpret)
    monkeypatch.setattr(comparison_api, "node_screens", lambda exp, run_id: [screen])
    interpret._read_screens()
    _screens_read(qapp, interpret)
    item = interpret.filters().screens.list.item(0)
    item.setCheckState(Qt.CheckState.Checked)
    assert interpret.filters().selection().leaves_source == screen.id


def test_a_label_written_while_a_screen_is_ticked_records_the_screen(
    qapp, fitted, monkeypatch
):
    """The origin reaches the run through the page's one write path, reads back from the
    state, and the label bar says it: the screen, how it read, and the labelled cluster's
    relation to the cluster the screen kept. A child of the kept cluster records *inside* it;
    with the screen unticked a label records nothing."""
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    node = _kept_candidate(view)
    _tick_screen(
        qapp, interpret, monkeypatch, _FakeScreen("cmp_07", [node], clearing=3)
    )

    _label(qapp, interpret, node, "climb", handle)
    origin = handle.state().label_origins[str(node)]
    assert origin == {
        "comparison": "cmp_07",
        "kept": str(node),
        "relation": "kept",
        "column": "drug",
        "screen": "drug vs vehicle · paired",
        "text": "more under drug than vehicle, 4.0× (10 of 10 recording(s) in that direction)",
        # Read at the screen's own floor: no level, and the kept cluster's numbers (§22).
        "alpha": "",
        "family_p": "0.02",
        "own_p": "0.004",
        "q": "0.5",
    }
    bar = interpret.label_bar()
    assert bar.origin.isVisibleTo(interpret)
    assert bar.origin.text() == (
        "From screen cmp_07 (drug vs vehicle · paired): more under drug than vehicle, "
        "4.0× (10 of 10 recording(s) in that direction) · kept · family-wise p 0.02, "
        "own p 0.004, q 0.5"
    )

    children = list(view.children(node))
    if children:
        child = int(children[0])
        _label(qapp, interpret, child, "climb left", handle)
        inside = handle.state().label_origins[str(child)]
        assert inside["relation"] == "under" and inside["kept"] == str(node)
        assert f"inside kept cluster #{node} ·" in bar.origin.text()

    interpret.filters().field_row("Differences").reset_button.click()
    other = next(
        n for n in range(view.tree.n_nodes - 1) if n != node and n not in children
    )
    _label(qapp, interpret, other, "rear", handle)
    assert str(other) not in handle.state().label_origins
    interpret.select(other)
    assert not bar.origin.isVisibleTo(interpret)


def test_undo_and_redo_carry_the_origin_with_the_label(qapp, fitted, monkeypatch):
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    node = _kept_candidate(interpret._view)
    _tick_screen(
        qapp, interpret, monkeypatch, _FakeScreen("cmp_07", [node], clearing=3)
    )
    _label(qapp, interpret, node, "climb", handle)
    interpret.undo()
    assert _drain(qapp, lambda: not handle.state().node_labels)
    assert _drain(qapp, lambda: not interpret.writes_pending())
    assert handle.state().label_origins == {}
    assert str(node) not in interpret._origins
    interpret.redo()
    assert _drain(
        qapp,
        lambda: handle.state().label_origins.get(str(node), {}).get("comparison")
        == "cmp_07",
    )
    assert _drain(qapp, lambda: not interpret.writes_pending())
    assert interpret._origins[str(node)]["comparison"] == "cmp_07"
    assert interpret.label_bar().origin.text().startswith("From screen cmp_07")


def test_the_origin_line_reads_the_stored_origin():
    from detail.gui.node_workspace import origin_line

    assert origin_line(None) == ""
    assert origin_line({}) == ""
    stored = {
        "comparison": "cmp_01",
        "kept": "1375",
        "relation": "under",
        "column": "drug",
        "screen": "drug vs vehicle · paired",
        "text": "more under drug than vehicle, 12.4×",
    }
    assert origin_line(stored) == (
        "From screen cmp_01 (drug vs vehicle · paired): more under drug than vehicle, "
        "12.4× · inside kept cluster #1375"
    )
    assert origin_line({**stored, "relation": "above", "text": ""}) == (
        "From screen cmp_01 (drug vs vehicle · paired) · above kept cluster #1375"
    )
    # Read looser than the screen's floor, with the kept cluster's numbers (§22); a level
    # that is the floor itself, or a blank, says nothing; a number that will not parse is
    # left out rather than fatal.
    numbered = {
        **stored,
        "relation": "kept",
        "alpha": "0.2",
        "family_p": "0.14",
        "own_p": "0.0039",
        "q": "0.9",
    }
    assert origin_line(numbered) == (
        "From screen cmp_01 (drug vs vehicle · paired): more under drug than vehicle, "
        "12.4× · kept at p ≤ 0.2 · family-wise p 0.14, own p 0.0039, q 0.9"
    )
    assert origin_line({**numbered, "alpha": "0.05", "q": "x"}) == (
        "From screen cmp_01 (drug vs vehicle · paired): more under drug than vehicle, "
        "12.4× · kept · family-wise p 0.14, own p 0.0039"
    )


def test_a_screen_read_looser_than_its_floor_shows_the_lead_and_the_label_records_it(
    qapp, fitted, monkeypatch
):
    """The slider at p ≤ 0.2 (§22): the dock's leaves grow by the lead the floor hid, the
    tree colours it and its hover says *kept at p ≤ 0.2* with its p's, the key's floor tick
    moves to that level, and a label written on it records the level and the numbers — a
    lead stays a lead for as long as the label exists. Back at the floor it is grey again.
    """
    from detail.analysis.compare import STRICTNESS_LEVELS

    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    node = _kept_candidate(view)
    lead = next(
        n
        for n in range(view.tree.n_nodes - 1)
        if n != node
        and n not in view.ancestors(node)
        and node not in view.ancestors(n)
        and view.summary(n).parent >= 0
    )
    _tick_screen(
        qapp,
        interpret,
        monkeypatch,
        _FakeScreen("cmp_12", [node], clearing=2, leads=[lead]),
    )
    bar = interpret.filters()
    assert bar.selection().leaves == view.leaves_under_nodes([node])
    assert bar.screens.strictness_label.text().endswith("1 kept of 2 clearing")
    assert interpret._icicle._diverging
    assert interpret._icicle.shades()[lead] == 0.0, "inside the floor: grey"

    bar.screens.strictness.setValue(STRICTNESS_LEVELS.index(0.2))
    selection = bar.selection()
    assert selection.alpha == 0.2
    assert selection.leaves == view.leaves_under_nodes([node, lead])
    assert bar.screens.strictness_label.text().endswith("2 kept of 3 clearing")
    assert interpret._icicle._diverging
    assert interpret._icicle.shades()[lead] > 0.0, "coloured at this level"
    assert "kept at p ≤ 0.2" in interpret._icicle._describe(lead)
    assert "family-wise p 0.2, own p 0.03, q 0.6" in interpret._icicle._describe(lead)
    interpret.select(lead)
    key = bar.dock.key
    assert key.is_effect() and "selected 2.5×" in key.text() and "floor" in key.text()

    _label(qapp, interpret, lead, "lead", handle)
    origin = handle.state().label_origins[str(lead)]
    assert origin["relation"] == "kept" and origin["kept"] == str(lead)
    assert origin["alpha"] == "0.2" and origin["family_p"] == "0.2"
    assert origin["own_p"] == "0.03" and origin["q"] == "0.6"
    assert (
        interpret.label_bar()
        .origin.text()
        .endswith("· kept at p ≤ 0.2 · family-wise p 0.2, own p 0.03, q 0.6")
    )

    bar.screens.strictness.setValue(STRICTNESS_LEVELS.index(0.05))
    assert bar.selection().alpha is None
    assert bar.selection().leaves == view.leaves_under_nodes([node])
    assert interpret._icicle.shades()[lead] == 0.0


def test_labels_queued_when_the_page_is_left_still_reach_the_run(
    qapp, fitted, monkeypatch
):
    """Three labels typed quickly, then the page left: all three reach the run (issue #24,
    finding 1).

    INTERPRET shows a label at once and writes it in the background, one write at a time, each
    reprojecting the run. The queue used to live on the page and each next write started from
    the previous one's ``done`` — a delivery ``GuiWorker`` drops once its receiver is gone — so
    leaving the page (the shell ``deleteLater``\\ s it) lost every write after the one in flight.
    Measured on main with a slowed write: three names on screen, one on disk. The queue now
    belongs to the run (``gui/run_writes.py``) and outlives the page.
    """
    import threading

    gate = threading.Event()
    _experiment, handle, workspace = _workspace(qapp, fitted)
    interpret = workspace._interpret
    view = interpret._view
    top = view.children(view.root)
    nodes = [*top[:2], view.children(top[0])[0]]
    real = type(handle).name_nodes

    def slow(self, *args, **kwargs):
        gate.wait(20)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(type(handle), "name_nodes", slow)
    names = {int(node): f"behavior{i}" for i, node in enumerate(nodes)}
    try:
        for node, name in names.items():
            interpret.select(node)
            interpret._label_field.edit.setText(name)
            interpret._label_field.submit()
        assert interpret.writes_pending()
        workspace.deleteLater()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    finally:
        gate.set()

    def on_disk() -> dict:
        state = handle.state().node_names
        return {node: state.get(str(node)) for node in names}

    assert _drain(qapp, lambda: on_disk() == names, 30_000), f"on disk: {on_disk()}"
