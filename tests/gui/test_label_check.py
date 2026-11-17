"""AUDIT on a clustering run: the list of labels and clusters, and one cluster's review (§19, §22).

What a reviewer does (review rounds 11 and 13): pick a cluster in the list, scroll through every
clip of it, click the wrong ones to remove them, then finish. Finishing writes one census and shows the
cluster's exact precision; a label's figure shows once every cluster under it is reviewed. No
verdicts anywhere. The run is the session's real temporal fit with three labels on its tree:
``climb``, ``climb/left`` nested in it, and ``rear``.

Classes these guard (review of 2026-11-09):

- **Two copies of one fact, one refreshed only on reload.** The open cluster's removals were rebuilt
  from the run's take-outs as last *read*, and only a reload re-read them; a click updated the open
  copy alone. Leaving a cluster and coming back showed its removals as kept, and a reload from
  INTERPRET landing mid-queue put back clips still being written. Now one copy, updated optimistically
  by each mark, with queued marks laid back over any read that lands before them.
- **The page's rule drifting from the backend's.** Every edit on a clip key showed as a removal; the
  backend counts only one made from the cluster's label or above (``_not_its_label``).
- **A fixture that cannot reach the path.** The least-typical-first check filtered out NaN
  typicality — and the session's fit is a medoid order whose node cache has *only* NaN, so it checked
  nothing while the page reviewed in time order.
- **A background read taking the keyboard.** A reload opened a cluster and focused the hidden grid.
"""

import threading

import numpy as np
import pytest

from detail.gui.label_check import (
    FROZEN,
    KEY_ROLE,
    LIST_MIN_PX,
    LIST_PX,
    LIST_SPLIT_KEY,
    ROW_ROLE,
    LabelCheckTab,
)
from detail.gui.label_clean import CLICK_HINT, SCROLL_FIRST
from detail.storage import audit_figures
from detail.storage import clip_review as review_store

from tests.gui.conftest import drain_busy_calls
from tests.gui.test_query_workspace import _drain


def _labelled(experiment, run_id):
    handle = experiment.temporal_clustering_run(run_id)
    view = handle.node_view()
    a, b = view.children(view.root)[:2]
    left = view.children(a)[0]
    handle.name_nodes({a: "climb", left: "left", b: "rear"})
    return handle, {"climb": a, "left": left, "rear": b}


@pytest.fixture
def labelled(qapp, fitted_temporal):
    experiment, run_id = fitted_temporal
    handle, nodes = _labelled(experiment, run_id)
    return experiment, handle, nodes


def _tab(qapp, experiment, handle):
    widget = LabelCheckTab(experiment, handle)
    widget.resize(1400, 900)
    widget.show()
    assert _drain(qapp, lambda: widget._loaded)
    drain_busy_calls(qapp)
    _ranked(qapp, widget)
    return widget


def _ranked(qapp, tab) -> None:
    """The fit's medoid order ranks a cluster's clips off the UI thread when it opens."""
    assert _drain(qapp, lambda: tab._open is None or not tab.pane._waiting, 20_000)


@pytest.fixture
def tab(qapp, labelled):
    experiment, handle, _nodes = labelled
    widget = _tab(qapp, experiment, handle)
    yield widget
    widget.pane.montage._worker.stop()
    widget.close()
    widget.deleteLater()


def _rows(tab) -> dict:
    """``{key: item}`` for every row of the list."""
    return {tuple(item.data(0, KEY_ROLE)): item for item in tab._items()}


def _look(tab, key) -> dict:
    return _rows(tab)[key].data(0, ROW_ROLE)


def _settle(qapp, tab) -> None:
    """Every write done, then every read that waited on them."""
    assert _drain(qapp, lambda: not tab.writes_pending(), 20_000)
    drain_busy_calls(qapp)


def _open(qapp, tab, path, node):
    tab._list.setCurrentItem(_rows(tab)[("cluster", path, node)])
    qapp.processEvents()
    assert tab._open["node"] == node
    _ranked(qapp, tab)


def _to_end(qapp, tab) -> None:
    """Scroll to the bottom until the pane says so: under load the grid's maximum can still
    grow after a fixed number of pumps, leaving the bar short of the real end."""
    bar = tab.pane.montage.grid.verticalScrollBar()

    def scrolled() -> bool:
        bar.setValue(bar.maximum())
        return tab.pane.reached_end()

    assert _drain(qapp, scrolled)


def _finish(qapp, tab) -> None:
    _to_end(qapp, tab)
    assert tab.pane.finish.isEnabled()
    tab.pane.finish.click()
    _settle(qapp, tab)
    assert _drain(qapp, lambda: tab.pane.is_finished())


# --- the list --------------------------------------------------------------------------------


def test_the_list_drags_wider_or_narrower_and_keeps_its_width(qapp, labelled):
    """A divider between the list and the clips, not a fixed column (review round 15): it
    stops at `LIST_MIN_PX`, and the width left is the one the next AUDIT opens with — read back
    through the real `QSettings`, which `qapp` points at this test process's own INI store, so
    the developer's settings are never read or written."""
    from PyQt6.QtCore import QSettings

    experiment, handle, _nodes = labelled
    first = _tab(qapp, experiment, handle)
    try:
        split = first._split
        fitted = first._list.width()
        assert fitted >= LIST_PX, "nothing stored yet: the default, fitted to its rows"
        assert fitted != 320, "the stored width below must differ from the fitted one"
        assert split.handle(1).isVisible(), "a handle to drag"
        split.moveSplitter(LIST_MIN_PX - 20, 1)
        assert first._list.width() == LIST_MIN_PX, "no narrower than the minimum"
        split.drag_handle(1, 40)  # past the minimum by the grip: the list shuts
        assert split.is_folded(0) and split.pane_name(0) == "Labels"
        split.drag_handle(1, 320)  # and drags back out
        assert not split.is_folded(0)
        assert first._list.width() == 320
        stored = QSettings().value(f"panes/{LIST_SPLIT_KEY}/sizes")
        assert str(stored).startswith("320,")
    finally:
        first.pane.montage._worker.stop()
        first.close()
        first.deleteLater()
    second = _tab(qapp, experiment, handle)
    try:
        assert second._list.width() == 320, "the next AUDIT opens at the width left"
    finally:
        second.pane.montage._worker.stop()
        second.close()
        second.deleteLater()


def test_the_list_holds_each_label_with_its_clusters_under_it(tab, labelled):
    _experiment, handle, nodes = labelled
    assert tab._pages.currentIndex() == tab.PAGE_REVIEW
    assert tab._list.width() >= LIST_PX
    rows = _rows(tab)
    climb = rows[("label", "climb")]
    assert [climb.child(i).data(0, KEY_ROLE) for i in range(climb.childCount())] == [
        ("cluster", "climb", nodes["climb"])
    ]
    assert (
        rows[("cluster", "climb/left", nodes["left"])].parent()
        is rows[("label", "climb/left")]
    )
    look = _look(tab, ("label", "climb"))
    assert (
        look["kind"] == "label" and look["colour"] and look["side"] == ""
    ), "no figure before every cluster is reviewed"
    cluster = _look(tab, ("cluster", "climb", nodes["climb"]))
    size = tab._cluster("climb", nodes["climb"])["size"]
    assert cluster["side"] == f"{size:,} clips"
    assert cluster["line"] in ("reviewing · 0 removed", "not reviewed")
    assert look["line"].endswith(
        "clips · 0 of 2 clusters reviewed"
    ), "climb takes in its sub-label"
    rear = _look(tab, ("cluster", "rear", nodes["rear"]))
    assert rear["line"] == "not reviewed" and rear["fraction"] == 0
    by_node = {r["node"]: r for r in handle.review_by_cluster()}
    assert (
        tab._cluster("climb", nodes["climb"])["size"] == by_node[nodes["climb"]]["size"]
    ), "the page's clips are the backend's review population"


def test_the_first_label_opens_its_first_cluster_every_clip_least_typical_first(
    tab, labelled
):
    _experiment, _handle, nodes = labelled
    pane = tab.pane
    assert tab._open["node"] == nodes["climb"]
    assert pane.title.text() == f"cluster #{nodes['climb']}"
    assert pane.label_name.text() == "climb"
    size = tab._cluster("climb", nodes["climb"])["size"]
    assert len(pane.montage.clips()) == size
    assert pane.finish_counts.text() == f"{size:,} climb · 0 not climb"
    assert pane.montage.is_clickable()
    typicality = tab._open["clips"].typicality[tab._open["order"]]
    # The fit's medoid order has no typicality in the node cache: every value NaN made the
    # order time order and this check empty, until the page ranked through `run.clips`.
    assert np.isfinite(typicality).all(), "ranked, not time order"
    assert list(typicality) == sorted(typicality, reverse=True), "least typical first"
    assert (
        _look(tab, ("cluster", "climb", nodes["climb"]))["line"]
        == "reviewing · 0 removed"
    )


def test_clicking_a_label_opens_its_first_unreviewed_cluster(qapp, tab, labelled):
    _experiment, _handle, nodes = labelled
    _click_row(qapp, tab, ("label", "rear"))
    assert tab._open["node"] == nodes["rear"]
    assert tab._list.currentItem().data(0, KEY_ROLE) == (
        "cluster",
        "rear",
        nodes["rear"],
    )


def test_open_in_interpret_is_offered_on_a_cluster(qapp, tab, labelled, monkeypatch):
    from PyQt6.QtWidgets import QMenu

    _experiment, _handle, nodes = labelled
    menus = []
    monkeypatch.setattr(QMenu, "popup", lambda self, _p: menus.append(self))
    asked = []
    tab.node_requested.connect(asked.append)
    item = _rows(tab)[("cluster", "rear", nodes["rear"])]
    tab._on_menu(tab._list.visualItemRect(item).center())
    (menu,) = menus
    (action,) = menu.actions()
    assert action.text() == "Open in INTERPRET"
    action.trigger()
    assert asked == [nodes["rear"]]


# --- removing ----------------------------------------------------------------------------------


def _click(tab, index, shift=False):
    """Through the montage's own click path, so a Shift-click runs from its real anchor."""
    from PyQt6.QtCore import Qt

    modifier = Qt.KeyboardModifier.ShiftModifier if shift else None
    tab.pane.montage._clicked(index, modifier)


def test_a_click_fades_the_tile_at_once_and_writes_off_the_ui_thread(
    qapp, tab, labelled, monkeypatch
):
    _experiment, handle, _nodes = labelled
    montage = tab.pane.montage
    seen = []
    real = type(handle).take_out_clips

    def recorded(self, *args, **kwargs):
        seen.append(threading.current_thread() is threading.main_thread())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(type(handle), "take_out_clips", recorded)
    _click(tab, 0)
    _click(tab, 2, shift=True)
    assert [i for i, c in enumerate(montage.clips()) if c.faded] == [0, 1, 2]
    first = montage.clips()[0]
    assert first.tag == "not climb" and first.crossed
    assert first.note.startswith("Removed from climb in review")
    _settle(qapp, tab)
    assert seen == [False, False], "written on a worker"
    assert len(handle.taken_out("climb")) == 3
    assert tab.pane.finish_counts.text().endswith(" · 3 not climb")
    assert tab.pane.undo_text.text() == "Removed 2 from climb", "the run was one step"
    look = _look(tab, ("cluster", "climb", tab._open["node"]))
    assert look["line"] == "reviewing · 3 removed"


def test_a_second_click_puts_the_clip_back(qapp, tab, labelled):
    _experiment, handle, _nodes = labelled
    montage = tab.pane.montage
    _click(tab, 0)
    _click(tab, 1)
    _settle(qapp, tab)
    _click(tab, 0)
    assert tab.pane.undo_text.text() == "Put back 1 in climb"
    assert [i for i, c in enumerate(montage.clips()) if c.faded] == [1]
    _settle(qapp, tab)
    assert len(handle.taken_out("climb")) == 1


def test_right_click_moves_a_clip_to_another_label(qapp, tab, labelled):
    _experiment, handle, _nodes = labelled
    menu = tab.pane.clip_menu(0)
    actions = {a.text(): a for a in menu.actions()}
    assert "Move to climb" not in actions and "Move to rear" in actions
    actions["Move to rear"].trigger()
    moved = tab.pane.montage.clips()[0]
    assert moved.tag == "→ rear" and moved.faded and not moved.crossed
    assert tab.pane.undo_text.text() == "Moved 1 to rear"
    texts = [a.text() for a in tab.pane.clip_menu(0).actions() if not a.isSeparator()]
    assert texts[-1] == "Put back in climb"
    assert "Move to rear" not in texts, "not offered the label it is already moved to"
    _settle(qapp, tab)
    (edit,) = handle.taken_out("climb")
    assert edit["to_name"] == "rear"


def _marks(tab) -> dict:
    return {i: c.tag for i, c in enumerate(tab.pane.montage.clips()) if c.faded}


def test_undo_reverses_each_kind_of_change_and_steps_back_several(qapp, tab, labelled):
    """Remove, put back, move: ⌘Z undoes them newest first, through the same ordered writes."""
    _experiment, handle, _nodes = labelled
    pane = tab.pane
    undo = {k.key().toString(): k for k in pane._keys}["Ctrl+Z"]
    _click(tab, 0)
    _click(tab, 1, shift=True)  # 0, 1 removed
    _click(tab, 1)  # 1 back
    move = next(a for a in pane.clip_menu(2).actions() if a.text().startswith("Move"))
    target = move.text().removeprefix("Move to ")
    move.trigger()  # 2 moved
    assert pane.undo_text.text() == f"Moved 1 to {target}"
    assert _marks(tab) == {0: "not climb", 2: f"→ {target}"}
    undo.activated.emit()
    assert _marks(tab) == {0: "not climb"}, "the move undone"
    assert pane.undo_text.text() == "Put back 1 in climb"
    undo.activated.emit()
    assert _marks(tab) == {0: "not climb", 1: "not climb"}
    pane.undo.click()
    undo.activated.emit()
    assert _marks(tab) == {}
    assert not pane.undo.isEnabled()
    assert pane.undo_text.text() == CLICK_HINT
    _settle(qapp, tab)
    assert handle.taken_out("climb") == [], "the run agrees once the writes land"


def test_a_double_click_opens_the_clip_and_removes_nothing(qapp, tab, monkeypatch):
    """The press before the double-click removed the clip; the pane takes it back."""
    from detail.gui import clip_popup

    opened = []
    monkeypatch.setattr(
        clip_popup,
        "open_clip_window",
        lambda opener, experiment, clip, *, index=0, **_kw: opened.append(index),
    )
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    canvas = tab.pane.montage.canvas
    at = tab.pane.montage.tile_rect(2).center()
    left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    # What a pointer sends: press, release, then the second press arrives as a double-click.
    QTest.mousePress(canvas, left, none, at)
    QTest.mouseRelease(canvas, left, none, at)
    assert _marks(tab) == {2: "not climb"}
    QTest.mouseDClick(canvas, left, none, at)
    QTest.mouseRelease(canvas, left, none, at)
    assert _marks(tab) == {}, "retracted"
    assert _drain(qapp, lambda: opened)
    assert opened == [2]
    _settle(qapp, tab)


def test_space_right_after_a_click_opens_the_clip_and_keeps_the_removal(
    qapp, tab, monkeypatch
):
    """Space opens the clip too, and the pane read any open inside the double-click interval
    as a double-click: a click then Space took the removal back."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    from detail.gui import clip_popup

    opened = []
    monkeypatch.setattr(
        clip_popup,
        "open_clip_window",
        lambda opener, experiment, clip, *, index=0, **_kw: opened.append(index),
    )
    _tap(qapp, tab, 2)
    QTest.keyClick(tab.pane.montage.canvas, Qt.Key.Key_Space)
    assert _drain(qapp, lambda: opened)
    assert opened == [2]
    assert _marks(tab) == {2: "not climb"}, "the click stands"
    _settle(qapp, tab)


def test_switching_cluster_clears_the_history(qapp, tab, labelled):
    _experiment, _handle, nodes = labelled
    _click(tab, 0)
    _open(qapp, tab, "rear", nodes["rear"])
    assert not tab.pane.undo.isEnabled()
    assert tab.pane.undo_text.text() == CLICK_HINT
    _settle(qapp, tab)


def test_a_click_keeps_the_scroll(qapp, tab):
    montage = tab.pane.montage
    montage.set_tile_px(
        240
    )  # the largest tiles in a small window, so a few clips scroll
    tab.resize(700, 520)
    for _ in range(5):
        qapp.processEvents()
    bar = montage.grid.verticalScrollBar()
    if bar.maximum() == 0:
        pytest.skip("this fit's cluster has too few clips to scroll")
    bar.setValue(bar.maximum())
    index = montage.visible_indices()[-1]
    _click(tab, index)
    assert bar.value() == bar.maximum() and bar.value() > 0
    _settle(qapp, tab)
    assert bar.value() == bar.maximum(), "the read-back after the write keeps it too"


# --- finishing ---------------------------------------------------------------------------------


def test_finish_waits_for_the_end_of_the_scroll(qapp, tab):
    montage = tab.pane.montage
    montage.set_tile_px(240)
    tab.resize(700, 460)
    for _ in range(5):
        qapp.processEvents()
    tab.open_cluster(
        tab._open["path"], tab._open["node"]
    )  # from the top, in the small window
    # The clips rank off the UI thread; until they arrive the grid is empty and "reached the
    # end" says nothing. A fixed five pumps lost that race in a loaded full run.
    _ranked(qapp, tab)
    for _ in range(5):
        qapp.processEvents()
    if tab.pane.reached_end():
        pytest.skip("this fit's cluster fits on one screen")
    assert not tab.pane.finish.isEnabled()
    assert tab.pane.finish.toolTip() == SCROLL_FIRST
    tab._finish()
    assert not tab.writes_pending(), "no finish before the end"
    _to_end(qapp, tab)
    assert tab.pane.finish.isEnabled()


def test_finish_writes_one_census_and_shows_the_exact_precision(
    qapp, tab, labelled, monkeypatch
):
    _experiment, handle, nodes = labelled
    calls = []
    real = type(handle).finish_review

    def recorded(self, node):
        calls.append((node, threading.current_thread() is threading.main_thread()))
        return real(self, node)

    monkeypatch.setattr(type(handle), "finish_review", recorded)
    _click(tab, 0)
    _click(tab, 1)
    size = len(tab.pane.montage.clips())
    assert tab.pane.finish_counts.text() == f"{size - 2:,} climb · 2 not climb"
    _finish(qapp, tab)
    assert calls == [(nodes["climb"], False)], "one finish, on a worker"
    log = review_store.read_review(handle._run_dir())
    assert len(log) == size and sum(r.answer == "no" for r in log) == 2
    expected = round(100 * (size - 2) / size)
    assert tab.pane.precision.text() == f"{expected}%"
    assert tab.pane.result.text() == f"{size - 2:,} of {size:,} are climb · 2 removed"
    assert tab.pane.when.text().startswith("· "), "the day it was reviewed"
    look = _look(tab, ("cluster", "climb", nodes["climb"]))
    assert look["line"] == f"{expected}% · 2 removed" and look["fraction"] == 1.0
    assert look["state"] == "reviewed"
    assert not tab.pane.undo.isEnabled(), "Finish clears the history"
    assert tab.pane.again.isVisible() and not tab.pane.finish.isVisible()


def test_a_finished_cluster_splits_its_clips_and_is_read_only(qapp, tab, labelled):
    _experiment, handle, _nodes = labelled
    montage = tab.pane.montage
    size = len(montage.clips())
    review_order = list(tab._open["display"])
    assert size >= 4, "the fixture's cluster has room for two removed and two kept"
    _click(tab, 1)
    _click(tab, 3)
    _finish(qapp, tab)
    assert montage.sections() == [
        ("Removed (2)", 2),
        (f"Kept as climb ({size - 2})", size - 2),
    ]
    shown = tab._open["display"]
    assert shown[:2] == [
        review_order[1],
        review_order[3],
    ], "removed first, in review order"
    assert [c.faded for c in montage.clips()[:3]] == [True, True, False]
    assert not montage.is_clickable()
    _click(tab, 2)
    assert not tab.writes_pending() and len(handle.taken_out("climb")) == 2
    assert tab._hover(0).startswith("#1 · "), "the tooltip follows the order shown"
    assert tab.pane.again.toolTip() == (
        "Change this cluster's removals, then finish again to update "
        f"{tab.pane.precision.text()}. Your 2 removals are kept."
    )
    tab.pane.again.click()
    assert tab._open["display"] == review_order, "Reopen review: the review order"
    assert len(handle.taken_out("climb")) == 2, "reopening keeps the removals"
    assert montage.sections() == [] and montage.is_clickable()


def test_a_finished_cluster_with_nothing_removed_shows_only_kept(qapp, tab):
    size = len(tab.pane.montage.clips())
    _finish(qapp, tab)
    assert tab.pane.montage.sections() == [(f"Kept as climb ({size})", size)]


def test_next_cluster_opens_the_next_unreviewed_one(qapp, tab, labelled):
    _experiment, _handle, nodes = labelled
    _finish(qapp, tab)
    assert tab.pane.next.text() == f"Next cluster: #{nodes['left']}"
    tab.pane.next.click()
    qapp.processEvents()
    assert tab._open["node"] == nodes["left"]
    assert not tab.pane.is_finished()


def test_a_label_shows_its_figure_only_once_every_cluster_under_it_is_reviewed(
    qapp, tab, labelled
):
    """``climb`` takes in ``climb/left``'s cluster too, so one finish is not enough."""
    _experiment, _handle, nodes = labelled
    _finish(qapp, tab)
    assert _look(tab, ("label", "climb"))["side"] == ""
    assert "1 of 2 clusters reviewed" in _rows(tab)[("label", "climb")].toolTip(0)
    assert _look(tab, ("label", "climb"))["line"].endswith("· 1 of 2 clusters reviewed")
    _open(qapp, tab, "climb/left", nodes["left"])
    _finish(qapp, tab)
    assert _look(tab, ("label", "climb"))["side"] == "100%"
    assert _look(tab, ("label", "climb/left"))["side"] == "100%"
    assert _look(tab, ("label", "rear"))["side"] == ""


def test_a_finished_cluster_opened_later_shows_its_result_and_reopen_review(
    qapp, tab, labelled
):
    _experiment, _handle, nodes = labelled
    _finish(qapp, tab)
    _open(qapp, tab, "rear", nodes["rear"])
    _open(qapp, tab, "climb", nodes["climb"])
    assert tab.pane.is_finished()
    assert tab.pane.box.isVisible() and tab.pane.precision.text() == "100%"
    assert tab.pane.again.toolTip().endswith("update 100%. Nothing was removed.")
    tab.pane.again.click()
    assert not tab.pane.is_finished() and tab.pane.montage.is_clickable()
    assert (
        _look(tab, ("cluster", "climb", nodes["climb"]))["line"]
        == "was 100% · 0 removed"
    ), "the old figure, said as out of date until Finish"


def test_the_earlier_one_by_one_audit_stays_in_the_tooltip(qapp, labelled):
    experiment, handle, _nodes = labelled
    for row in handle.audit_sample(path="climb", budget=10):
        handle.label_audit(row.instance_id, True)
    widget = _tab(qapp, experiment, handle)
    try:
        tip = _rows(widget)[("label", "climb")].toolTip(0)
        assert "Earlier one-by-one audit: precision" in tip
        assert _look(widget, ("label", "climb"))["side"] == "", "never merged"
    finally:
        widget.pane.montage._worker.stop()
        widget.deleteLater()


def test_a_snapshot_shows_every_clip_but_marks_and_finishes_nothing(qapp, labelled):
    experiment, handle, _nodes = labelled
    snapshot = handle.view(handle.pin(reason="export"))
    widget = _tab(qapp, experiment, snapshot)
    try:
        pane = widget.pane
        assert pane.montage.clips(), "watching works"
        _to_end(qapp, widget)
        assert not pane.finish.isEnabled() and pane.finish.toolTip() == FROZEN
        assert not pane.montage.is_clickable()
        assert all(not a.isEnabled() for a in pane.clip_menu(0).actions())
        pane.montage.clip_clicked.emit(0, False)
        widget._take_out([0], None)
        widget._finish()
        assert not widget.writes_pending()
    finally:
        widget.pane.montage._worker.stop()
        widget.deleteLater()


# --- what a figure shows ---------------------------------------------------------------------


def test_an_edited_label_says_how_many_judgments_came_before():
    """The reported text: "not audited yet · … · edited after the audit" for a label with 78."""
    assert audit_figures.notes({"n": 0, "since": "t", "n_before": 78}) == [
        "edited after the audit: 0 audited since, 78 before"
    ]


def test_with_nothing_left_to_audit_the_share_is_exact():
    """Every clip judged is a count, not a sample: the exact share, no interval to narrow."""
    six = {"n": 6, "precision": 5 / 6, "interval": (0.44, 0.97), "remaining": 0}
    assert audit_figures.cells(six) == ("0.83", "all audited")
    assert audit_figures.line(six) == "precision 0.83  (all 6 audited)"
    assert audit_figures.complete(six)
    unknown = {k: v for k, v in six.items() if k != "remaining"}
    assert not audit_figures.complete(unknown), "never assumed complete"


def _row_bar_run(qapp, look) -> int:
    """The longest horizontal run of the bar's colours on the second line of a cluster row drawn
    with ``look`` — a bar is a long run, a glyph's edge is not."""
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem

    from detail.gui import plot_theme
    from detail.gui.label_check import ReviewListDelegate

    tree = QTreeWidget()
    tree.setHeaderHidden(True)
    tree.setItemDelegate(ReviewListDelegate(tree))
    tree.resize(LIST_PX, 120)
    item = QTreeWidgetItem(tree)
    item.setData(0, ROW_ROLE, look)
    tree.show()
    qapp.processEvents()
    rect = tree.visualItemRect(item)
    image = tree.viewport().grab().toImage()
    theme = plot_theme.current()
    bar = {QColor(theme.grid).name(), QColor(theme.summary).name()}
    second_top = rect.top() + rect.height() // 2
    longest = 0
    for y in range(second_top, rect.bottom()):
        run = 0
        for x in range(rect.left(), rect.right()):
            run = run + 1 if image.pixelColor(x, y).name() in bar else 0
            longest = max(longest, run)
    tree.deleteLater()
    return longest


def test_a_reviewed_cluster_row_has_no_bar(qapp):
    """Round 14: "✓ Reviewed" and the figure say it all; a full bar beside them said nothing.
    A row still in review keeps its bar."""
    look = {
        "kind": "cluster",
        "name": "cluster #871",
        "side": "62 clips",
        "fraction": 0.5,
    }
    reviewing = dict(look, state="reviewing", line="reviewing · 2 removed")
    reviewed = dict(look, state="reviewed", line="97% · 2 removed", fraction=1.0)
    assert _row_bar_run(qapp, reviewing) > 40
    assert (
        _row_bar_run(qapp, reviewed) < 8
    ), "the narrowest bar the old row drew was 12px"


def _widest_row(tab) -> int:
    from PyQt6.QtWidgets import QStyleOptionViewItem

    lst = tab._list
    option = QStyleOptionViewItem()
    option.initFrom(lst)
    return max(
        lst.itemDelegate().row_width(option, lst.indexFromItem(item))
        + (0 if item.parent() is None else lst.indentation())
        for item in tab._items()
    )


def test_the_list_opens_wide_enough_that_no_row_is_cut(qapp, tab):
    """A fixed 230px cut every label's second line at 1440x847 — "182 clips · 0 of 3 clusters
    review…". Until it is dragged, the list opens as wide as its widest row."""
    tab.resize(1400, 800)
    tab.show()
    qapp.processEvents()
    try:
        assert (
            _widest_row(tab) > LIST_PX - 30
        ), "the fixture's rows must be near the old width"
        assert tab._list.viewport().width() >= _widest_row(tab), (
            tab._list.viewport().width(),
            _widest_row(tab),
        )
    finally:
        tab.hide()


def test_a_long_label_opens_the_list_no_wider_than_its_cap(qapp, tab):
    """However long a name, the list opens at `LIST_MAX_LINES` at most: past it the row elides,
    its whole text on hover."""
    from detail.gui.label_check import LIST_MAX_LINES
    from detail.gui.sizing import lines

    tab.resize(1400, 800)
    tab.show()
    qapp.processEvents()
    try:
        label = next(item for item in tab._items() if item.parent() is None)
        look = dict(label.data(0, ROW_ROLE), name="a very long label name " * 6)
        label.setData(0, ROW_ROLE, look)
        tab._fit_list_width()
        qapp.processEvents()
        assert tab._list.width() == lines(LIST_MAX_LINES)
        assert label.toolTip(0), "an elided row says the rest on hover"
    finally:
        tab.hide()


# --- one copy of what is removed, the list's keys, focus (review of 2026-11-09) ----------------


def _tap(qapp, tab, index) -> None:
    """A real left click on tile ``index`` of the grid."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    montage = tab.pane.montage
    QTest.mouseClick(
        montage.canvas,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        montage.tile_rect(index).center(),
    )
    qapp.processEvents()


def _click_row(qapp, tab, key) -> None:
    """A real left click on the list's row ``key``."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    item = _rows(tab)[key]
    tab._list.scrollToItem(item)
    qapp.processEvents()
    QTest.mouseClick(
        tab._list.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        tab._list.visualItemRect(item).center(),
    )
    qapp.processEvents()
    _ranked(qapp, tab)


def test_leaving_a_cluster_and_coming_back_keeps_its_removals(qapp, tab, labelled):
    """Removals made by click were held only in the open cluster; coming back rebuilt them
    from the run as first read, so they showed as kept — and Finish then counted them
    removed. Once with the writes landed, once with them still queued."""
    _experiment, _handle, nodes = labelled
    rear = ("cluster", "rear", nodes["rear"])
    climb = ("cluster", "climb", nodes["climb"])
    _click_row(qapp, tab, rear)
    assert tab._open["node"] == nodes["rear"]
    _tap(qapp, tab, 0)
    _tap(qapp, tab, 1)
    _settle(qapp, tab)
    marks, counts = _marks(tab), tab.pane.finish_counts.text()
    assert len(marks) == 2
    _click_row(qapp, tab, climb)
    _tap(qapp, tab, 0)  # left with this one still being written
    _click_row(qapp, tab, rear)
    assert _marks(tab) == marks
    assert tab.pane.finish_counts.text() == counts
    assert _look(tab, rear)["line"] == "reviewing · 2 removed"
    _click_row(qapp, tab, climb)
    assert len(_marks(tab)) == 1, "the mark made just before leaving"
    _click_row(qapp, tab, rear)
    total = len(tab.pane.montage.clips())
    _finish(qapp, tab)
    assert tab.pane.result.text() == f"{total - 2:,} of {total:,} are rear · 2 removed"


def test_a_reopened_review_left_and_returned_to_keeps_its_removals_and_old_figure(
    qapp, tab, labelled
):
    """After Reopen review the old figure stood as current, in the box and the list, and
    coming back showed the cluster finished with its new removals unseen."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    _experiment, _handle, nodes = labelled
    climb = ("cluster", "climb", nodes["climb"])
    _finish(qapp, tab)
    old = tab.pane.precision.text()
    QTest.mouseClick(tab.pane.again, Qt.MouseButton.LeftButton)
    assert tab.pane.was.text() == f"was {old} · reviewing again"
    _tap(qapp, tab, 0)
    _tap(qapp, tab, 1)
    assert _look(tab, climb)["line"] == f"was {old} · 2 removed"
    _click_row(qapp, tab, ("cluster", "rear", nodes["rear"]))
    assert _look(tab, climb)["line"] == f"was {old} · 2 removed"
    assert _look(tab, climb)["state"] == "reviewing"
    _click_row(qapp, tab, climb)
    assert not tab.pane.is_finished(), "still being reviewed again"
    assert len(_marks(tab)) == 2
    assert not tab.pane.was.isHidden() and old in tab.pane.was.text()
    total = len(tab.pane.montage.clips())
    _finish(qapp, tab)
    assert tab.pane.result.text() == f"{total - 2:,} of {total:,} are climb · 2 removed"
    assert tab.pane.was.isHidden()
    assert _look(tab, climb)["line"] == f"{tab.pane.precision.text()} · 2 removed"


def test_a_read_landing_mid_queue_keeps_the_marks_still_being_written(
    qapp, tab, labelled, monkeypatch
):
    """INTERPRET's ``changed`` reloads AUDIT whenever it likes; a read that lands before the
    queued writes must not put their clips back on screen."""
    _experiment, handle, _nodes = labelled
    gate = threading.Event()
    real = type(handle).take_out_clips

    def slow(self, *args, **kwargs):
        gate.wait(20)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(type(handle), "take_out_clips", slow)
    try:
        _tap(qapp, tab, 0)
        _tap(qapp, tab, 1)
        assert tab.writes_pending()
        marks = _marks(tab)
        rendered = []
        real_render = tab._render
        tab._render = lambda data, **kw: (real_render(data, **kw), rendered.append(1))
        tab.reload()
        assert _drain(qapp, lambda: rendered)
        assert tab.writes_pending(), "the read landed before the writes"
        assert _marks(tab) == marks
        assert tab._open["edits"].keys() == {
            tab._key_of(r) for r in tab._rows_at(sorted(marks))
        }
    finally:
        gate.set()
    _settle(qapp, tab)
    assert _marks(tab) == marks
    assert len(handle.taken_out("climb")) == 2


def test_an_edit_made_under_another_label_is_not_a_removal_here(qapp, labelled):
    """The backend counts an edit only when it was made from the cluster's label or one above
    it (``_not_its_label``); the page showed every edit on the clip as a removal."""
    experiment, handle, nodes = labelled
    view = handle.node_view()
    rows = view.clips(nodes["rear"]).rows()[:2]
    handle.take_out_clips(rows, "climb", None)  # made under another label
    widget = _tab(qapp, experiment, handle)
    try:
        _click_row(qapp, widget, ("cluster", "rear", nodes["rear"]))
        assert _marks(widget) == {}
        assert (
            _look(widget, ("cluster", "rear", nodes["rear"]))["line"]
            == "reviewing · 0 removed"
        )
        assert "0 removed" in _rows(widget)[("cluster", "rear", nodes["rear"])].toolTip(
            0
        )
        total = len(widget.pane.montage.clips())
        _finish(qapp, widget)
        assert (
            widget.pane.result.text() == f"{total:,} of {total:,} are rear · 0 removed"
        )
    finally:
        widget.pane.montage._worker.stop()
        widget.deleteLater()


def test_the_arrow_keys_move_through_the_list_past_a_label(qapp, tab, labelled):
    """A label row redirected the current row to its first cluster, so Up from that cluster
    landed back on it: the arrows could never pass a label."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    _experiment, _handle, nodes = labelled
    order = [tuple(i.data(0, KEY_ROLE)) for i in tab._items()]
    tab._list.setFocus()
    tab._list.setCurrentItem(_rows(tab)[order[-1]])
    qapp.processEvents()
    seen = [order[-1]]
    for _ in range(len(order) - 1):
        QTest.keyClick(tab._list, Qt.Key.Key_Up)
        qapp.processEvents()
        seen.append(tuple(tab._list.currentItem().data(0, KEY_ROLE)))
    assert seen == order[::-1], "every row in turn, labels included"
    assert tab._list.hasFocus(), "the keyboard stays in the list"
    for _ in range(order.index(("cluster", "climb/left", nodes["left"]))):
        QTest.keyClick(tab._list, Qt.Key.Key_Down)
    qapp.processEvents()
    assert tab._open["node"] == nodes["left"], "a cluster the arrows reach opens"


def test_a_label_row_clicked_or_entered_opens_its_first_unreviewed_cluster(
    qapp, tab, labelled
):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    _experiment, _handle, nodes = labelled
    _click_row(qapp, tab, ("label", "rear"))
    assert tab._open["node"] == nodes["rear"]
    assert tuple(tab._list.currentItem().data(0, KEY_ROLE)) == (
        "cluster",
        "rear",
        nodes["rear"],
    )
    assert tab.pane.montage.canvas.hasFocus(), "a click hands the clips the keyboard"
    tab._list.setFocus()
    tab._list.setCurrentItem(_rows(tab)[("label", "climb/left")])
    qapp.processEvents()
    assert tab._open["node"] == nodes["rear"], "passing a label opens nothing"
    QTest.keyClick(tab._list, Qt.Key.Key_Return)
    qapp.processEvents()
    assert tab._open["node"] == nodes["left"]


def test_a_label_tooltip_counts_the_same_clusters_as_its_row(qapp, tab, labelled):
    """The row's "N clips" takes in sub-labels' clusters; the tooltip's "x of y clips
    removed" counted only the label's own."""
    _experiment, _handle, nodes = labelled
    _click_row(qapp, tab, ("cluster", "climb/left", nodes["left"]))
    _tap(qapp, tab, 0)
    _settle(qapp, tab)
    line = _look(tab, ("label", "climb"))["line"]
    clips = line.split(" clips")[0]
    tip = _rows(tab)[("label", "climb")].toolTip(0)
    assert f"1 of {clips} clips removed" in tip.splitlines()


def test_a_method_that_ranks_on_demand_is_reviewed_least_typical_first(
    qapp, labelled, monkeypatch
):
    """dtw_finch's medoid order has no typicality in the node cache: the page reviewed its
    clips in time order. It ranks them as INTERPRET does — ``run.clips`` off the UI thread —
    saying so meanwhile."""
    from detail.gui.label_clean import RANKING

    experiment, handle, nodes = labelled
    assert handle.node_view().ranks_on_demand, "the session's fit is a medoid order"
    gate = threading.Event()
    asked = []

    def ranked(self, node, progress=None):
        asked.append(threading.current_thread() is threading.main_thread())
        gate.wait(20)
        clips = self.node_view().clips(int(node))
        return clips.with_typicality(np.arange(len(clips), dtype=np.float64))

    monkeypatch.setattr(type(handle), "clips", ranked)
    widget = LabelCheckTab(experiment, handle)
    widget.resize(1400, 900)
    widget.show()
    try:
        assert _drain(qapp, lambda: asked)
        assert widget.pane.undo_text.text() == RANKING
        assert not widget.pane.can_mark()
        assert not widget.pane.finish.isEnabled()
        gate.set()
        assert _drain(qapp, lambda: len(widget.pane.montage.clips()) > 0)
        assert asked[0] is False, "ranked on a worker"
        typicality = widget._open["clips"].typicality[widget._open["order"]]
        assert list(typicality) == sorted(typicality, reverse=True)
        assert widget.pane.can_mark()
    finally:
        gate.set()
        widget.pane.montage._worker.stop()
        widget.deleteLater()


def test_a_reload_in_a_hidden_tab_takes_no_focus(qapp, labelled):
    """INTERPRET's label writes reload AUDIT while it is hidden; opening its cluster took the
    keyboard into the hidden grid, and INTERPRET's keys stopped working until a click.
    """
    from PyQt6.QtWidgets import QLineEdit, QTabWidget

    experiment, handle, _nodes = labelled
    tabs = QTabWidget()
    field = QLineEdit()
    tabs.addTab(field, "INTERPRET")
    widget = LabelCheckTab(experiment, handle)
    tabs.addTab(widget, "AUDIT")
    tabs.resize(1200, 800)
    tabs.show()
    try:
        field.setFocus()
        assert _drain(qapp, lambda: widget._loaded)
        drain_busy_calls(qapp)
        widget.reload()
        drain_busy_calls(qapp)
        qapp.processEvents()
        assert widget.focusWidget() is None, "nothing inside the hidden tab holds focus"
        assert tabs.focusWidget() is field
    finally:
        widget.pane.montage._worker.stop()
        tabs.deleteLater()


def test_marks_queued_when_the_page_is_left_still_reach_the_run(
    qapp, labelled, monkeypatch
):
    """Leaving AUDIT with writes queued must not lose them (issue #24, finding 1).

    The queue used to live on the page: each next write started from the previous write's
    ``done``, and that delivery is dropped once the page is gone — so every mark after the
    first in flight was lost without a word when the shell tore the page down. The queue now
    belongs to the run (``gui/run_writes.py``) and finishes after the page has gone.
    """
    from PyQt6.QtCore import QEvent

    experiment, handle, _nodes = labelled
    gate = threading.Event()
    real = type(handle).take_out_clips

    def slow(self, *args, **kwargs):
        gate.wait(20)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(type(handle), "take_out_clips", slow)
    widget = _tab(qapp, experiment, handle)
    try:
        for index in range(3):
            _tap(qapp, widget, index)
        assert len(_marks(widget)) == 3
        assert widget.writes_pending()
        widget.pane.montage._worker.stop()
        widget.close()
        widget.deleteLater()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    finally:
        gate.set()
    assert _drain(
        qapp, lambda: len(handle.taken_out("climb")) == 3, 20_000
    ), f"only {len(handle.taken_out('climb'))} of 3 marks reached the run"
