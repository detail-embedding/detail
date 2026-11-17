"""A cluster's pop-out window beside INTERPRET (``gui/cluster_window.py``, the 11-9 meeting).

**What is pinned.** One window per cluster: opening a cluster that has one raises it. Labelling in
a window goes through the page's own queue, so the run, the tree and the page's bar agree, and
the window's title follows — and INTERPRET's selection stays where it was (D1 B); each window's
number marks its cluster on the tree. Keys 1–9 label the window's cluster, not the page's. Enter,
sent as a real key, only ever submits the field (``test_enter_*``). One ⌘Z history for the page and
its windows (``test_undo_*``). A re-fit closes every window. A snapshot greys the row with the
page's reason. The
window's montage toolbar is every clip grid's (``test_montage_export.py``); here, that its picks
export in pick order with the window's layers, orientation and context. The run is the session's
kinematics fit (``fitted``).
"""

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QInputDialog, QMessageBox

from detail.gui.cluster_workspace import ClusterWorkspace


def _drain(qapp, predicate, timeout_ms: int = 10_000) -> bool:
    from tests.gui.conftest import drain_until

    return drain_until(qapp, predicate, timeout_ms)


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    raised: list[str] = []

    def critical(parent, title, text, *args, **kwargs):
        raised.append(f"{title}: {text}")
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "critical", staticmethod(critical))
    yield
    assert not raised, f"a background call failed: {raised}"


def _interpret(qapp, fitted, handle=None):
    experiment, run_id = fitted
    handle = handle or experiment.authored_run(run_id)
    workspace = ClusterWorkspace(experiment, handle)
    interpret = workspace._interpret
    assert _drain(qapp, lambda: interpret._view is not None), "INTERPRET never loaded"
    # A bout method ranks a node's clips off the UI thread on first ask
    # (`NodeView.ranks_on_demand`), so the opening node's montage lands after the view does.
    assert _drain(
        qapp, lambda: interpret._clips is not None
    ), "the opening node's clips never arrived"
    return experiment, handle, workspace, interpret


def _two_clusters(view):
    """Two clusters with several clips each, neither inside the other (so either may take a
    label the other has)."""
    nodes = sorted(range(view.tree.n_nodes), key=lambda n: (-len(view.clips(n)), -n))
    found = [n for n in nodes if n != view.root and len(view.clips(n)) >= 3]
    a = found[0]
    for b in found[1:]:
        if a not in view.ancestors(b) and b not in view.ancestors(a):
            return a, b
    raise AssertionError("the fit has no two separate clusters of three clips")


def _quiet(qapp, interpret, timeout_ms: int = 20_000) -> bool:
    """Every queued label write has landed."""
    return _drain(
        qapp,
        lambda: not interpret.writes_pending(),
        timeout_ms,
    )


def _window(qapp, interpret, node):
    window = interpret.open_window(node)
    assert _drain(qapp, lambda: window._clips is not None)
    return window


def test_one_window_per_cluster_and_a_second_open_raises_the_first(qapp, fitted):
    _e, _h, workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    first = _window(qapp, interpret, a)
    assert not first.isModal() and first.isVisible()
    assert interpret.open_window(a) is first, "never two windows for one cluster"
    other = _window(qapp, interpret, b)
    assert other is not first
    assert {w.node for w in interpret.windows().windows()} == {a, b}
    first.close()
    qapp.processEvents()
    assert [w.node for w in interpret.windows().windows()] == [b]
    again = _window(qapp, interpret, a)
    assert again is not first, "a closed window is opened afresh"
    workspace.deleteLater()
    assert _drain(qapp, lambda: sip.isdeleted(other)), "closes with the page"


def test_the_montage_header_and_the_tree_menu_open_the_window(qapp, fitted):
    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    interpret.select(a)
    interpret._open_window.click()
    assert interpret.windows().get(a) is not None
    menu = interpret._icicle.context_menu(a)
    texts = [action.text() for action in menu.actions()]
    assert "Open in window" in texts
    interpret._icicle.window_wanted.emit(a)
    assert len(interpret.windows().windows()) == 1


def test_labelling_in_a_window_reaches_the_run_and_the_page(qapp, fitted):
    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    interpret.select(b)
    window = _window(qapp, interpret, a)
    window.bar.field.edit.setText("rear")
    window.bar.field.submit()
    assert interpret._names.get(str(a)) == "rear", "the page holds it at once"
    assert window.windowTitle().startswith(f"① {interpret._cluster_name(a)} · rear · ")
    assert window.bar.chip.text() == "rear"
    assert interpret._selected == b, "INTERPRET stays where it was"
    assert interpret._icicle._selected == b
    assert _drain(qapp, lambda: handle.node_names().get(str(a)) == "rear")


def test_number_keys_in_a_window_label_the_windows_cluster(qapp, fitted):
    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    interpret._label(b, "climb")  # now a recent label, key 1
    interpret.select(b)
    window = _window(qapp, interpret, a)
    assert window.bar.quick.names()[0] == "climb"
    window.montage.canvas.setFocus()
    QTest.keyClick(window.montage.canvas, Qt.Key.Key_1)
    assert interpret._names.get(str(a)) == "climb"
    assert _drain(qapp, lambda: handle.node_names().get(str(a)) == "climb")
    entries = window.bar.quick.actions()
    assert [action.text() for action in entries] == ["1  climb"]
    assert not entries[0].isEnabled(), "this cluster is “climb” already"
    # No Recent ▾ in the window: the numbered entries lead its Labels ▾, as on the page.
    listed = window.bar.field.labels_menu().actions()
    assert listed[1].text() == "1  climb" and not listed[1].isEnabled()


def test_a_labels_menu_choice_in_a_window_fills_its_field_and_enter_writes(
    qapp, fitted
):
    """The window's bar is the page's (``build_label_bar``): a click on the field types, a
    Labels ▾ choice only fills the field — nothing reaches the run — and Enter writes it.
    """
    from PyQt6.QtCore import QPoint

    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    interpret._label(b, "climb")
    assert _quiet(qapp, interpret)
    window = _window(qapp, interpret, a)
    field = window.bar.field
    field.edit.setText("")
    QTest.mouseClick(field.edit, Qt.MouseButton.LeftButton, pos=QPoint(4, 4))
    qapp.processEvents()
    assert qapp.activePopupWidget() is None, "a click in the field opened a list"
    menu = field.labels_menu()
    row = next(x for x in menu.actions() if x.data() == "climb" and x.menu())
    menu.popup(QPoint(0, 0))
    qapp.processEvents()
    QTest.mouseClick(
        menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(row).center()
    )
    qapp.processEvents()
    assert field.edit.text() == "climb"
    assert str(a) not in interpret._names and _quiet(qapp, interpret)
    assert str(a) not in handle.node_names(), "a menu choice wrote to the run"
    QTest.keyClick(field.edit, Qt.Key.Key_Return)
    assert interpret._names.get(str(a)) == "climb"
    assert _drain(qapp, lambda: handle.node_names().get(str(a)) == "climb")
    assert _quiet(qapp, interpret)


def test_typing_a_number_in_the_field_labels_nothing(qapp, fitted):
    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    interpret._label(b, "climb")
    window = _window(qapp, interpret, a)
    window.bar.field.edit.setFocus()
    QTest.keyClick(window.bar.field.edit, Qt.Key.Key_1)
    assert window.bar.field.edit.text().endswith("1"), "the key typed into the field"
    assert not interpret._names.get(str(a))


def test_focus_on_a_window_leaves_interpret_where_it_was(qapp, fitted, monkeypatch):
    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    interpret.select(b)
    monkeypatch.setattr(window, "isActiveWindow", lambda: True)
    window.changeEvent(QEvent(QEvent.Type.ActivationChange))
    assert interpret._selected == b
    assert interpret._icicle._selected == b


def test_each_window_numbers_its_cluster_on_the_tree(qapp, fitted):
    """D1 B: ① ② on the tree, the same number in each window's title; a closed window's number
    is the next one handed out, and its mark goes."""
    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    view = interpret._view
    a, b = _two_clusters(view)
    first = _window(qapp, interpret, a)
    second = _window(qapp, interpret, b)
    icicle = interpret._icicle
    assert icicle.window_marks() == {a: 1, b: 2}
    assert first.windowTitle().startswith("① ") and second.windowTitle().startswith(
        "② "
    )
    third = next(n for n in range(view.tree.n_nodes) if n not in (a, b, view.root))
    interpret.select(third)
    assert (
        icicle._selected == third
    ), "INTERPRET keeps its own selection beside the marks"
    first.close()
    qapp.processEvents()
    assert icicle.window_marks() == {b: 2}
    again = _window(qapp, interpret, a)
    assert again.number == 1 and icicle.window_marks() == {a: 1, b: 2}
    assert "open in window 1" in icicle._describe(a)


@pytest.mark.parametrize("how", ["escape", "close button", "title bar"])
def test_every_way_a_window_closes_lets_it_go(qapp, fitted, how):
    """Reported in review: Esc closes a ``QDialog`` through ``done()``, which sends no close
    event, so a window closed by Esc kept its mark on the tree, its place in the page's list
    (and with it its montage and tile cache) and never saved where it was. Esc is sent as a
    real key to the grid, where the focus sits."""
    from PyQt6.QtCore import QByteArray, QSettings

    from detail.gui.cluster_window import GEOMETRY_KEY

    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    first, second = _window(qapp, interpret, a), _window(qapp, interpret, b)
    qapp.processEvents()
    QSettings().remove(GEOMETRY_KEY)
    if how == "escape":
        first.activateWindow()
        first.montage.canvas.setFocus()
        QTest.keyClick(first.montage.canvas, Qt.Key.Key_Escape)
    elif how == "close button":
        from PyQt6.QtWidgets import QDialogButtonBox

        box = first.findChild(QDialogButtonBox)
        box.button(QDialogButtonBox.StandardButton.Close).click()
    else:
        first.close()
    # Before the event loop runs: nothing may wait for the deferred delete to prune it.
    assert interpret._icicle.window_marks() == {b: 2}
    assert list(interpret._windows._open) == [b]
    assert first.montage.source() is None, "its tiles are let go"
    assert isinstance(QSettings().value(GEOMETRY_KEY), QByteArray)
    assert second.isVisible()


def test_finish_while_labels_are_saving_waits_for_them_and_locks_every_label_surface(
    qapp, fitted, monkeypatch
):
    """Reported in review: **Finish** clicked while a label write was still queued ran the
    finalize beside the write queue — "frame views … out of date after an edit", the run left
    open — and the windows stayed editable meanwhile. Finish now waits for the writes, and from
    the click until the run is finished nothing labels: not the page, not a window, not a key,
    not ⌘Z."""
    import time

    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    run = interpret._run
    read = run.label_maps

    def slow():
        time.sleep(0.4)
        return read()

    monkeypatch.setattr(run, "label_maps", slow)
    interpret._label(b, "rear")
    assert interpret.writes_pending(), "the write is still in flight"
    interpret._finish.click()
    reason = interpret._edit_reason()
    assert reason.startswith("The run is being finished")
    assert not interpret._finish.isEnabled()
    for bar in (interpret._label_bar, window.bar):
        assert not bar.field.edit.isEnabled() and bar.field.edit.toolTip() == reason
        assert not bar.undo.isEnabled() and bar.undo.toolTip() == reason
        assert not bar.quick.action(0).isEnabled()
        assert bar.quick.action(0).toolTip() == reason
    window.montage.canvas.setFocus()
    QTest.keyClick(window.montage.canvas, Qt.Key.Key_1)
    interpret.undo()
    assert interpret._names.get(str(a)) is None, "key 1 labelled nothing"
    assert interpret._names.get(str(b)) == "rear", "⌘Z took nothing back"
    assert _drain(qapp, lambda: handle.status == "complete", 30_000)
    assert _drain(qapp, lambda: interpret._reopen.isEnabled())
    assert handle.node_names().get(str(b)) == "rear"
    assert handle.node_names().get(str(a)) is None
    finished = interpret._edit_reason()
    assert finished.startswith("This run is finished")
    assert window.bar.field.edit.toolTip() == finished


def test_a_write_that_failed_leaves_nothing_to_undo_on_screen(
    qapp, fitted, monkeypatch
):
    """Review: after a failed write the history is cleared, but ⌘Z stayed live on the page and in
    every window until something else re-synced it."""
    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    said: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda _p, title, text, *a, **k: said.append(title)),
    )

    def refuse():
        raise OSError("the share went away")

    monkeypatch.setattr(interpret._run, "label_maps", refuse)
    interpret._label(b, "rear")
    assert interpret._label_bar.undo.isEnabled(), "live while the write is in flight"
    assert _drain(qapp, lambda: said == ["Label not saved"])
    assert not interpret._history.can_undo()
    assert not interpret._label_bar.undo.isEnabled()
    assert not window.bar.undo.isEnabled()


def test_a_drawn_mark_sits_inside_its_block_off_the_selection_ring(qapp, fitted):
    _e, _h, workspace, interpret = _interpret(qapp, fitted)
    view = interpret._view
    icicle = interpret._icicle
    top = list(view.children(view.root))
    interpret.select(top[0])
    _window(qapp, interpret, top[0])
    workspace.resize(1440, 847)
    workspace.show()
    qapp.processEvents()
    icicle.grab()
    rects = icicle.window_mark_rects()
    assert set(rects) == {top[0]}, "the widest block wears its window's number"
    drawn = {n: r for r, n in icicle._blocks}
    block = max(
        (drawn[n] for n in icicle._drawn_as(top[0], drawn)), key=lambda r: r.width()
    )
    disc = rects[top[0]]
    assert block.adjusted(3, 3, -3, -3).contains(disc), "inside the selection ring"


def test_a_snapshot_greys_the_windows_label_row_with_the_reason(qapp, fitted):
    experiment, run_id = fitted
    live = experiment.authored_run(run_id)
    view = live.node_view()
    first, _second = view.children(view.root)
    live.name_nodes({first: "pinned"})
    snapshot_id = live.pin(reason="finish")
    _e, _h, _workspace, interpret = _interpret(
        qapp, fitted, handle=live.view(snapshot_id)
    )
    window = _window(qapp, interpret, first)
    reason = interpret._edit_reason()
    assert "snapshot" in reason
    field = window.bar.field
    for widget in (field.edit, field.label_button, field.remove_button):
        assert not widget.isEnabled() and widget.toolTip() == reason
    window.bar.field.edit.setText("other")
    window.bar.field.submit()
    assert interpret._names.get(str(first)) == "pinned", "nothing was written"
    # The page says the same reason (it said "This run cannot be edited here."), and the tree's
    # right-click greys Label… and Remove label with it rather than doing nothing silently.
    page = interpret._label_bar.field
    for widget in (page.edit, page.label_button, page.remove_button):
        assert not widget.isEnabled() and widget.toolTip() == reason
    menu = interpret._icicle.context_menu(first)
    entries = {action.text(): action for action in menu.actions()}
    for text in ("Label…", "Remove label"):
        assert not entries[text].isEnabled() and entries[text].toolTip() == reason


# --- the montage toolbar: pick mode and the export ------------------------------------------ #
def test_the_windows_picks_export_in_pick_order_with_its_look_and_context(
    qapp, fitted, monkeypatch, tmp_path
):
    """The window's export is every clip grid's (`test_montage_export.py`); this pins that the
    window hands it its own clips, in its own order, and that its **Show ▾** reaches the job.
    """
    from detail.api import clip_montage
    from detail.gui import animation_export

    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    order = window.display_order()
    montage = window.montage
    exports = montage.exports
    assert not montage.is_clickable(), "outside pick mode a click does nothing here"
    exports.start_picking()
    montage._clicked(2, Qt.KeyboardModifier.NoModifier)
    montage._clicked(1, Qt.KeyboardModifier.NoModifier)
    assert not montage.show_button.layer_actions[
        "keypoints"
    ].isEnabled(), "the fit read no keypoints"
    montage.show_button.layer_actions["footage"].trigger()
    montage.show_button.context_actions[1.0].trigger()
    job = exports.job()
    view = interpret._view.clips(a)
    second = 1_000_000_000
    assert [(c.start_t_ns, c.end_t_ns) for c in job.clips] == [
        (max(int(view.start_ns[i]) - second, 0), int(view.end_ns[i]) + second)
        for i in (order[2], order[1])
    ], "the picks, in pick order, widened by the context"
    assert job.labels[0].startswith("#3 ") and job.labels[1].startswith("#2 ")
    assert not job.layers.raw and job.layers.mask and not job.layers.aligned
    line = exports.button.describe_action.text()
    assert "1 s context" in line and "as recorded" in line

    seen = {}

    class _Montage:
        def __init__(self, experiment_dir, clips_):
            seen["clips"] = clips_

        def export(self, path, layers_, labels_, **kw):
            seen.update(layers=layers_, labels=labels_, path=path)
            return path

        def close(self):
            pass

    monkeypatch.setattr(clip_montage, "ClusterMontage", _Montage)
    monkeypatch.setattr(
        animation_export,
        "run_export",
        lambda target, work, *, on_written, label: on_written(work(None)),
    )
    exports.write_montage(job, tmp_path / "picks.mp4")
    assert seen["clips"] == job.clips and seen["layers"] == job.layers
    assert seen["labels"] == job.labels
    assert "picks.mp4" in exports.written.text()


def test_clips_removed_in_audit_are_faded_and_tagged_in_the_window(qapp, fitted):
    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    clips = window._clips
    i = window.display_order()[0]
    key = interpret._clip_key_of(clips, i)
    interpret._taken = {key: {"from": "climb", "to": None, "at": "2026-11-08"}}
    interpret._windows.refresh_marks()
    tile = window.montage.clips()[0]
    assert tile.faded and tile.tag


def test_an_audit_change_reaches_a_window_while_the_pages_grid_is_empty(qapp, fitted):
    """Review: the page re-marked its windows only after its own early returns, so a removal
    in AUDIT while the page's grid was empty or re-ranking never reached an open window.
    """
    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    i = window.display_order()[0]
    key = interpret._clip_key_of(window._clips, i)
    interpret.montage().set_clips([])  # the page's grid, between two clusters
    interpret._taken = {key: {"from": "climb", "to": None, "at": "2026-11-08"}}
    interpret._mark_taken()
    assert window.montage.clips()[0].faded


def test_a_double_click_on_the_page_opens_the_clip_with_the_montages_context(
    qapp, fitted, monkeypatch
):
    """Review: the page opened the clip window with one second of context whatever **Show ▾**
    said; a cluster window passed its own."""
    from detail.gui import clip_popup

    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    asked: list[float] = []
    monkeypatch.setattr(
        clip_popup,
        "open_clip_window",
        lambda *a, **k: asked.append(k["context_s"]),
    )
    montage = interpret.montage()
    for seconds in (0.0, 2.0):
        montage.set_context_s(seconds)
        montage.clip_activated.emit(0)
    assert asked == [0.0, 2.0]


def test_a_reload_of_the_same_tree_keeps_the_windows_and_the_history(qapp, fitted):
    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    first, second = _window(qapp, interpret, a), _window(qapp, interpret, b)
    interpret._label(a, "rear", first)
    assert _quiet(qapp, interpret)
    interpret._view = None
    interpret.reload()
    assert _drain(qapp, lambda: interpret._view is not None)
    qapp.processEvents()
    assert first.isVisible() and second.isVisible()
    assert interpret._history.can_undo()


def test_a_refit_closes_every_window_and_says_so(qapp, fitted):
    """A re-fit renumbers the tree, so a window kept open would show another cluster under its
    old number: every window closes, whether or not its node id still exists, and ⌘Z forgets.
    """
    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    first, second = _window(qapp, interpret, a), _window(qapp, interpret, b)
    interpret._label(a, "rear", first)
    assert _quiet(qapp, interpret)
    interpret._tree_key = ("another fit",)  # what the loaded tree was before a re-fit
    interpret._view = None
    interpret.reload()
    assert _drain(qapp, lambda: interpret._view is not None)
    qapp.processEvents()
    assert all(sip.isdeleted(w) or not w.isVisible() for w in (first, second))
    assert interpret.windows().windows() == []
    assert interpret._written.text() == (
        "Closed 2 cluster windows: the run was re-fitted."
    )
    assert not interpret._history.can_undo()
    assert not interpret._label_bar.undo.isEnabled()


def test_the_tree_identity_follows_the_fit_and_the_merges(qapp, fitted):
    from detail.gui.node_workspace import tree_identity

    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    view = interpret._view
    assert tree_identity(view) == tree_identity(handle.node_view()), "a re-read"
    assert tree_identity(None) is None

    class _Renumbered:
        built_from = view.built_from
        tree = type(
            "T",
            (),
            {"children": view.tree.children[::-1], "n_leaves": view.tree.n_leaves},
        )

    assert tree_identity(_Renumbered) != tree_identity(view)


def test_the_window_greys_aligned_with_the_runs_reason(qapp, fitted):
    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    note = interpret._align_note
    assert note, "the kinematics fit read no keypoints"
    button = window.montage.show_button
    assert button.text() == "Show ▾", "the window's Show ▾ is every grid's"
    for actions in (button, interpret.montage().show_button):
        aligned = actions.orientation_actions[True]
        assert not aligned.isEnabled() and aligned.toolTip() == note


def test_new_previews_reach_an_open_window_and_keep_its_picks(qapp, fitted):
    """A window opened before the previews were built would keep showing "no preview": when the
    page's previews change, each window opens its tiles afresh, picks untouched."""
    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    exports = window.montage.exports
    exports.start_picking()
    window.montage._clicked(0, Qt.KeyboardModifier.NoModifier)
    picks, source = exports.picks(), window.montage.source()
    interpret._previews_arrived()
    assert _drain(qapp, lambda: window.montage.source() is not source)
    assert exports.picks() == picks


class _Flat:
    """A tile source that draws grey: pictures to keep, where the fixture has no previews."""

    def frame(self, _index, _offset_ns, tile_px, _matching, *, look):
        import numpy as np

        return np.full((tile_px, tile_px, 3), 128, np.uint8)


def test_a_recordings_masks_landing_moves_nothing_in_any_montage(qapp, fitted):
    """Review: a recording's masks finishing re-set the tiles of the page and of every open
    window — each window jumped back to its top, every clock restarted, and a method that ranks
    on demand ranked again. Now each montage reads its pictures afresh in place: the same tiles,
    the scroll and the clocks kept, the old pictures shown until the new ones land."""
    _e, _h, workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    window.montage.set_tile_px(48)
    window.resize(420, 300)
    workspace.show()
    qapp.processEvents()
    montages = {"window": window.montage, "INTERPRET": interpret.montage()}
    for montage in montages.values():
        montage.set_source(_Flat())  # the fixture builds no previews: pictures to keep
        assert _drain(qapp, lambda m=montage: len(m._images) > 0)
    bar = window.montage.grid.verticalScrollBar()
    assert (
        bar.maximum() > 0
    ), "the window's grid has to scroll for this to mean anything"
    bar.setValue(bar.maximum())
    before = {
        name: (
            m.grid.verticalScrollBar().value(),
            list(m._offsets),
            m.clips(),
            m.source(),
        )
        for name, m in montages.items()
    }
    seen = {name: _watched(m) for name, m in montages.items()}
    for montage in montages.values():
        montage.advance(0)  # nothing more: the clocks are stopped by _watched
    rid = window._clips.recording_ids[0]
    interpret._missing_masks = [rid]
    interpret._on_masks_done(rid)
    for name, montage in montages.items():
        scroll, offsets, clips, source = before[name]
        assert seen[name]["calls"] == ["set_source"], f"{name}: {seen[name]['calls']}"
        assert montage.source() is not source, f"{name} reads the new masks"
        assert montage.grid.verticalScrollBar().value() == scroll, f"{name} scrolled"
        assert montage._offsets == offsets, f"{name}'s clocks restarted"
        assert montage.clips() is clips or montage.clips() == clips
        assert montage._images, f"{name} blanked its tiles"


def test_a_reload_reads_each_windows_clips_once_and_keeps_its_place(
    qapp, fitted, monkeypatch
):
    """Review: a page reload read each window's clips twice (once as new previews, once as the
    reload) and sent it to its top both times."""
    _e, _h, workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    window.montage.set_tile_px(48)
    window.resize(420, 300)
    workspace.show()
    qapp.processEvents()
    bar = window.montage.grid.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(bar.maximum())
    scroll = bar.value()
    loads: list[int] = []
    real = window._show_clips
    monkeypatch.setattr(
        window, "_show_clips", lambda clips, **k: (loads.append(1), real(clips, **k))
    )
    interpret._view = None
    interpret.reload()
    assert _drain(qapp, lambda: interpret._view is not None)
    # This view ranks on demand, so the window's clips arrive from a worker thread: wait for
    # the read itself, not one pump after the page's view (that raced on a 2-core Linux
    # runner). Then give a second read half a second to show up, and require none.
    assert _drain(qapp, lambda: loads), "the reload never re-read the window's clips"
    assert not _drain(qapp, lambda: len(loads) > 1, timeout_ms=500), loads
    assert window.montage.grid.verticalScrollBar().value() == scroll


# --- Enter, sent as a real key ------------------------------------------------------------ #
# The bug these pin: a QDialog makes its first push button — Remove — the default, so Enter in
# the field labelled the cluster, then the field passed Return on and the dialog clicked Remove.
# The tests that missed it called `field.submit()` and `keyPressEvent` directly, which never go
# near the dialog's default-button handling; these send the key the way a person does.
@pytest.fixture
def _nothing_opens(monkeypatch):
    """Record a menu or a save dialog Enter opens instead of blocking on it: before the fix,
    Enter with Remove greyed clicked **Labels ▾** — the next default — and its menu waited.
    """
    from PyQt6.QtWidgets import QMenu

    from detail.gui import animation_export

    opened: list[str] = []
    monkeypatch.setattr(QMenu, "exec", lambda menu, *a, **k: opened.append("menu"))
    monkeypatch.setattr(
        animation_export,
        "ask_save_path",
        lambda *a, **k: opened.append("save dialog"),
    )
    return opened


def _labelled(qapp, interpret, handle, window, node, name):
    """``node`` labelled ``name`` through the window's field and Enter, landed on disk."""
    field = window.bar.field
    field.edit.setFocus()
    field.edit.selectAll()
    QTest.keyClicks(field.edit, name)
    QTest.keyClick(field.edit, Qt.Key.Key_Return)
    assert _quiet(qapp, interpret)
    qapp.processEvents()
    assert _quiet(qapp, interpret)


def test_enter_in_the_field_labels_the_cluster_and_keeps_the_label(
    qapp, fitted, _nothing_opens
):
    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    _labelled(qapp, interpret, handle, window, a, "rear")
    assert not _nothing_opens
    assert handle.node_names().get(str(a)) == "rear", "Enter labelled it and kept it"
    assert interpret._names.get(str(a)) == "rear"
    assert window.bar.field.remove_button.isEnabled(), "Remove is live, and was not hit"


@pytest.mark.parametrize("where", ["canvas", "remove", "quit", "window"])
def test_enter_anywhere_else_in_the_window_clicks_nothing(
    qapp, fitted, where, _nothing_opens
):
    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    _labelled(qapp, interpret, handle, window, a, "rear")
    exports = window.montage.exports
    exports.start_picking()
    picks = exports.picks()
    target = {
        "canvas": window.montage.canvas,
        "remove": window.bar.field.remove_button,
        "quit": exports.pick_bar.quit,
        "window": window,
    }[where]
    target.setFocus()
    for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
        QTest.keyClick(target, key)
    assert _quiet(qapp, interpret)
    assert handle.node_names().get(str(a)) == "rear", f"Enter on {where} removed it"
    assert exports.is_picking() and exports.picks() == picks and window.isVisible()
    assert not _nothing_opens, f"Enter on {where} opened {_nothing_opens}"


def test_no_button_in_the_window_is_default_or_auto_default(qapp, fitted):
    from PyQt6.QtWidgets import QPushButton

    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    armed = [
        b.text()
        for b in window.findChildren(QPushButton)
        if b.isDefault() or b.autoDefault()
    ]
    assert armed == []


# --- the window follows the page's status line, and asks over itself --------------------- #
def test_the_windows_status_line_follows_the_pages_to_output_up_to_date(qapp, fitted):
    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    _labelled(qapp, interpret, handle, window, a, "rear")
    assert _drain(qapp, lambda: "Output up to date" in window.bar.status.text())
    assert window.bar.status.text() == interpret._label_bar.status.text()


def test_rename_merge_and_a_refused_name_ask_over_the_window_that_asked(
    qapp, fitted, monkeypatch
):
    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    interpret._label(a, "rear")
    interpret._label(b, "climb")
    assert _quiet(qapp, interpret)
    window = _window(qapp, interpret, a)
    parents = {}

    def get_text(parent, *args, **kwargs):
        parents["rename"] = parent
        return "", False

    def get_item(parent, *args, **kwargs):
        parents["merge"] = parent
        return "", False

    def warning(parent, *args, **kwargs):
        parents["warning"] = parent
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(get_text))
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(get_item))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(warning))
    window.bar.field.rename_wanted.emit("rear")
    window.bar.field.merge_wanted.emit("rear")
    window.bar.field.edit.setText("a/b")
    window.bar.field.submit()
    assert parents == {"rename": window, "merge": window, "warning": window}
    parents.clear()
    interpret._label_field.rename_wanted.emit("rear")
    assert parents == {"rename": interpret}, "from the page, over the page"


# --- one ⌘Z history for the page and its windows ------------------------------------------ #
def _undo_key(target):
    from PyQt6.QtGui import QKeySequence

    seq = QKeySequence(QKeySequence.StandardKey.Undo)[0]
    QTest.keyClick(target, seq.key(), seq.keyboardModifiers())


def test_undo_in_a_window_puts_back_the_label_it_wrote(qapp, fitted):
    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    _labelled(qapp, interpret, handle, window, a, "rear")
    assert window.bar.undo.isEnabled()
    assert window.bar.undo.toolTip().startswith(f"Undo: Labelled #{a} “rear”")
    # The focus is still in the field: ⌘Z must reach the history, not the field's own undo.
    _undo_key(window.bar.field.edit)
    assert not interpret._names.get(str(a)), "shown at once"
    assert _quiet(qapp, interpret)
    assert not handle.node_names().get(str(a)), "and written"
    assert interpret._label_bar.status.text().startswith(f"Undid: Labelled #{a}")
    assert window.bar.status.text() == interpret._label_bar.status.text()


def test_undo_on_the_page_undoes_the_windows_change_and_redo_makes_it_again(
    qapp, fitted
):
    from PyQt6.QtGui import QKeySequence

    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    interpret._label(b, "climb")
    window = _window(qapp, interpret, a)
    _labelled(qapp, interpret, handle, window, a, "rear")
    interpret._icicle.setFocus()
    _undo_key(interpret._icicle)
    assert _quiet(qapp, interpret)
    assert not handle.node_names().get(str(a))
    assert handle.node_names().get(str(b)) == "climb", "one step back, not two"
    seq = QKeySequence(QKeySequence.StandardKey.Redo)[0]
    QTest.keyClick(window.montage.canvas, seq.key(), seq.keyboardModifiers())
    assert _quiet(qapp, interpret)
    assert handle.node_names().get(str(a)) == "rear", "redone from the window"


def test_undo_of_a_merge_brings_the_label_back_under_its_own_id(
    qapp, fitted, monkeypatch
):
    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    interpret._label(a, "rear")
    interpret._label(b, "climb")
    assert _quiet(qapp, interpret)
    before = handle.label_maps()
    monkeypatch.setattr(
        QInputDialog, "getItem", staticmethod(lambda *a, **k: ("climb", True))
    )
    interpret._merge_label("rear")
    assert _quiet(qapp, interpret)
    assert handle.node_names().get(str(a)) == "climb"
    interpret.undo()
    assert _quiet(qapp, interpret)
    assert handle.label_maps() == before, "the same ids, not a new label called rear"
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("rearing", True))
    )
    interpret._rename_label("rear")
    assert _quiet(qapp, interpret)
    interpret.undo()
    assert _quiet(qapp, interpret)
    assert handle.label_maps() == before


def test_undo_queued_behind_its_own_write_still_undoes_it(qapp, fitted):
    """⌘Z pressed before the write lands: the undo waits its turn in the queue and reads what
    the write did once it has."""
    _e, handle, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    interpret._label(a, "rear")
    interpret.undo()
    assert _quiet(qapp, interpret)
    assert not handle.node_names().get(str(a))


def test_undo_is_greyed_with_the_reason_on_a_snapshot(qapp, fitted):
    experiment, run_id = fitted
    live = experiment.authored_run(run_id)
    snapshot_id = live.pin(reason="finish")
    _e, _h, _workspace, interpret = _interpret(
        qapp, fitted, handle=live.view(snapshot_id)
    )
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    reason = interpret._edit_reason()
    for button in (interpret._label_bar.undo, window.bar.undo):
        assert not button.isEnabled() and button.toolTip() == reason


def test_nothing_to_undo_says_so(qapp, fitted):
    from detail.gui.label_history import NOTHING_TO_UNDO

    _e, _h, _workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    window = _window(qapp, interpret, a)
    for button in (interpret._label_bar.undo, window.bar.undo):
        assert not button.isEnabled() and button.toolTip() == NOTHING_TO_UNDO


# --- a label written in one window leaves every other montage's pictures alone ----------- #
def _watched(montage):
    """Count what would make ``montage`` re-lay or re-ask for pictures: new tiles, a new
    source, a cleared cache, and every frame request — its clock stopped, so playback asks for
    nothing of its own while it is watched."""
    montage._clock.stop()
    montage._sync_clock = lambda: None
    seen = {"calls": [], "posts": 0, "generation": montage._generation}
    for name in ("set_source", "set_clips", "_reset_pictures"):
        real = getattr(montage, name)

        def wrapped(*args, _real=real, _name=name, **kwargs):
            seen["calls"].append(_name)
            return _real(*args, **kwargs)

        setattr(montage, name, wrapped)
    post = montage._worker.post

    def counted(*args, **kwargs):
        seen["posts"] += 1
        return post(*args, **kwargs)

    montage._worker.post = counted
    return seen


def test_a_label_in_one_window_leaves_the_other_montages_pictures_alone(qapp, fitted):
    """Reported 2026-11-09: labelling in one window re-rendered the clips of every other. A
    label changes no tile, so neither the other window's montage nor INTERPRET's own may be
    handed new tiles, a new source, a cleared cache or a single frame request."""
    from tests.gui.conftest import drain_until

    _e, handle, workspace, interpret = _interpret(qapp, fitted)
    a, b = _two_clusters(interpret._view)
    first, second = _window(qapp, interpret, a), _window(qapp, interpret, b)
    workspace.resize(1440, 847)
    workspace.show()
    drain_until(qapp, lambda: False, 1000)
    other, page = _watched(second.montage), _watched(interpret.montage())
    _labelled(qapp, interpret, handle, first, a, "rear")
    drain_until(qapp, lambda: False, 1500)
    assert handle.node_names().get(str(a)) == "rear"
    for name, seen, montage in (
        ("the other window", other, second.montage),
        ("INTERPRET", page, interpret.montage()),
    ):
        assert seen["calls"] == [], f"{name} was handed {seen['calls']}"
        assert seen["posts"] == 0, f"{name} asked for {seen['posts']} frames"
        assert montage._generation == seen["generation"], f"{name}'s cache was cleared"
    assert second.windowTitle().startswith("② "), "its title still follows the page"


def test_marks_changed_sees_a_fade_and_ignores_the_filter_match():
    from detail.gui.live_montage import MontageClip
    from detail.gui.node_workspace import marks_changed

    tiles = [MontageClip(1, 1, "a"), MontageClip(1, 2, "b")]
    assert not marks_changed(tiles, list(tiles))
    assert not marks_changed(
        [MontageClip(1, 1, "a", matching=False), tiles[1]], tiles
    ), "the page sets the match itself"
    assert marks_changed(tiles, [MontageClip(1, 1, "a", faded=True), tiles[1]])
    assert marks_changed(tiles, tiles[:1])


# --- the window's top is INTERPRET's label bar ---------------------------------------------- #
#
# The class: two surfaces meant to look the same, built by two pieces of code. The window had a
# bespoke "compact label row" beside INTERPRET's `LabelBar`, and its montage toolbar on a row of
# its own, so the two drifted — three rows over the window's grid where the page had one bar. Both
# are built by `node_workspace.build_label_bar` now; this compares what each bar *holds*, row by
# row, and what sits between each bar and its grid, so a row added to either one fails here.


def _control(widget) -> str:
    """A bar's control as what it is: its class, and a button's words (digits dropped, so
    "Export ▾" and "Undo (⌘Z)" compare and a count does not)."""
    from PyQt6.QtWidgets import QAbstractButton

    name = type(widget).__name__
    if isinstance(widget, QAbstractButton):
        return f"{name}({''.join(c for c in widget.text() if not c.isdigit()).strip()})"
    return name


def bar_rows(bar, *, skip=()) -> list[list[str]]:
    """``bar``'s lines, top to bottom, each its controls left to right — every widget not
    deliberately hidden, through nested layouts and the montage's toolbar. ``skip``: widgets
    (and layouts) left out by identity."""
    from PyQt6.QtWidgets import QWidget

    def leaves(item) -> list:
        layout = item.layout() if hasattr(item, "layout") else None
        widget = item.widget() if hasattr(item, "widget") else item
        if widget is not None and isinstance(widget, QWidget):
            if widget in skip or widget.isHidden():
                return []
            inner = widget.layout()
            if inner is not None and widget.findChildren(QWidget):
                return leaves(inner)
            return [_control(widget)]
        if layout is not None:
            if layout in skip:
                return []
            out = []
            for i in range(layout.count()):
                out += leaves(layout.itemAt(i))
            return out
        return []

    top = bar.layout()
    rows = [leaves(top.itemAt(i)) for i in range(top.count())]
    return [row for row in rows if row]


def widgets_between(bar, grid) -> list[str]:
    """Visible widgets laid out between ``bar``'s bottom and ``grid``'s top, in their common
    window — a row someone put over the clips. Ancestors of the grid (the montage, the scroll
    area) and anything inside the bar are not rows."""
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QWidget

    window = bar.window()
    low = bar.mapTo(window, QPoint(0, bar.height())).y()
    high = grid.mapTo(window, QPoint(0, 0)).y()
    out = []
    for widget in window.findChildren(QWidget):
        if (
            not widget.isVisible()
            or widget.height() < 4
            or widget.isAncestorOf(grid)
            or bar.isAncestorOf(widget)
            or widget is bar
        ):
            continue
        top = widget.mapTo(window, QPoint(0, 0)).y()
        if low - 1 <= top and top + widget.height() <= high + 1:
            out.append(f"{type(widget).__name__}({widget.objectName()}) at {top}")
    return out


def test_the_windows_top_is_interprets_label_bar_row_for_row(qapp, fitted):
    from PyQt6.QtWidgets import QLabel, QPushButton

    from detail.gui.label_panel import LabelBar

    _e, _h, workspace, interpret = _interpret(qapp, fitted)
    a, _b = _two_clusters(interpret._view)
    workspace.resize(1440, 900)
    workspace.show()
    interpret.select(a)
    window = _window(qapp, interpret, a)
    _drain(qapp, lambda: False, 500)
    page_bar, bar = interpret.label_bar(), window.bar
    assert type(bar) is LabelBar and type(page_bar) is LabelBar
    page_rows = bar_rows(page_bar, skip=(interpret._open_window, page_bar.lines))
    assert bar_rows(bar, skip=(bar.lines,)) == page_rows
    assert page_rows[0][-1].startswith("ExportMenu"), "the toolbar ends line 1"
    assert [w for w in page_rows[2] if w.startswith("QPushButton")] == [
        "QPushButton(Label)",
        "QPushButton(Remove)",
        "QPushButton(Labels ▾)",
        "QPushButton(Undo (⌘Z))",
    ]
    assert "Open in window" not in {b.text() for b in bar.findChildren(QPushButton)}
    assert bar.name.text().startswith("① ") and bar.name.text().endswith(f"#{a}")
    assert bar.figures.text().split(" · ")[0].endswith(" clips")
    for host in (interpret, window):
        found = widgets_between(
            host.label_bar() if host is interpret else host.bar,
            host.montage().grid if host is interpret else host.montage.grid,
        )
        assert found == [], f"rows between the bar and the grid: {found}"

    # It can fail: a control added to one bar, or a row put over the window's grid.
    extra = QPushButton("Extra")
    bar.add_action_widget(extra)
    _drain(qapp, lambda: False, 100)
    assert bar_rows(bar, skip=(bar.lines,)) != page_rows
    extra.hide()
    row = QLabel("a row over the grid")
    window.body().insertWidget(1, row)
    _drain(qapp, lambda: False, 300)
    assert widgets_between(bar, window.montage.grid), "the planted row is caught"


def test_an_order_picked_while_the_window_ranks_draws_nothing_until_the_clips_land(
    qapp, fitted
):
    """The window empties its tiles while its clips rank, and its own clips with them: emptying
    only the tiles let the order and context switches draw the old clips back under
    "ranking…" (`node_workspace._clear_clips` is the page's half of this)."""
    _experiment, handle, _workspace, interpret = _interpret(qapp, fitted)
    node, _other = _two_clusters(interpret._view)
    window = _window(qapp, interpret, node)
    assert window.montage.clips(), "the window shows its clips"

    window.load_clips()
    assert window.montage.clips() == []
    assert window.display_order() == [], "the window's order agrees with its tiles"
    other = next(k for k in window.montage.order_actions if k != window.montage.order())
    window.montage.order_actions[other].trigger()
    assert window.montage.clips() == [], "the old tiles came back while ranking"
    window._on_context()
    assert window.montage.clips() == [], "the old tiles came back while ranking"

    assert _drain(qapp, lambda: window._clips is not None)
    assert window.display_order() == handle.clips(node).order(other, seed=node).tolist()
    assert len(window.montage.clips()) == len(window.display_order())
