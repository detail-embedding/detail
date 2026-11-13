"""One montage toolbar on every clip grid, and its pick mode (``gui/montage_export.py``, §22).

**The class: a capability added to one clip grid and not the others.** INTERPRET's montage, the
pop-out cluster window and AUDIT's review grid each grew their own view and export controls — a
Show row and a Context box and a picks bar in the window, *Show masks* on the page, the montage
exports in the shell's Export… for INTERPRET only — so the same clips were exported three ways and
AUDIT not at all. The toolbar is now built by ``live_montage.LiveMontage`` itself, ``export``
a required keyword, so a grid cannot be built without it; :func:`test_every_clip_grid_has_one_toolbar`
reads the three real grids and compares them action by action.

**What is pinned.** The toolbar's controls and their order; **Show ▾**'s context widening what an
export writes; **Export ▾** writing every clip in grid order with **Show ▾**'s layers (the job
handed to the writer, not real encoding); pick mode entered from **Export ▾**, left by **Quit** and
by Esc, with the grid's own click given back (AUDIT removes again) and nothing written to the run
by a pick; nothing picked on entry, each click numbering the next; the picks exported in pick
order; the bar's export greyed with the reason when nothing
is picked; the cluster window without its permanent rows; and the shell's Export… without montage
entries. The run is the session's temporal fit with ``climb`` and ``rear`` on its tree.
"""

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QSlider

from detail.gui.cluster_workspace import ClusterWorkspace
from detail.gui.montage_export import NOTHING_PICKED, ExportMenu
from detail.gui.show_menu import ShowButton

from tests.gui.conftest import drain_busy_calls, drain_until

NO_MODIFIER = Qt.KeyboardModifier.NoModifier


def _drain(qapp, predicate, timeout_ms: int = 20_000) -> bool:
    return drain_until(qapp, predicate, timeout_ms)


@pytest.fixture
def workspace(qapp, fitted_temporal):
    """The temporal fit's workspace, two clusters labelled, INTERPRET on its biggest one."""
    experiment, run_id = fitted_temporal
    handle = experiment.temporal_clustering_run(run_id)
    view = handle.node_view()
    a, b = view.children(view.root)[:2]
    handle.name_nodes({a: "climb", b: "rear"})
    widget = ClusterWorkspace(experiment, handle)
    widget.resize(1400, 900)
    widget.show()
    interpret, audit = widget._interpret, widget._audit
    assert _drain(qapp, lambda: interpret._view is not None and audit._loaded)
    drain_busy_calls(qapp)
    interpret.select(a)
    montage = interpret.montage()
    assert _drain(
        qapp,
        lambda: len(montage.clips()) > 2
        and len(montage.clips()) == len(interpret._order),
    )
    widget.nodes = {"climb": a, "rear": b}
    yield widget
    for window in interpret.windows().windows():
        window.close()
    widget.close()
    widget.deleteLater()


def _window(qapp, interpret, node):
    window = interpret.open_window(node)
    assert _drain(qapp, lambda: window._clips is not None and window.montage.clips())
    return window


def _audit(qapp, workspace):
    """AUDIT with ``climb``'s cluster open in its review pane."""
    audit = workspace._audit
    workspace._tabs.setCurrentWidget(audit)
    audit.open_cluster("climb", workspace.nodes["climb"])
    # The fit's medoid order ranks a cluster's clips off the UI thread when it opens.
    assert _drain(qapp, lambda: audit.pane.montage.clips())
    return audit


def _grids(qapp, workspace) -> dict:
    interpret = workspace._interpret
    window = _window(qapp, interpret, workspace.nodes["climb"])
    return {
        "INTERPRET": interpret.montage(),
        "cluster window": window.montage,
        "AUDIT": _audit(qapp, workspace).pane.montage,
    }


def _toolbar(montage) -> list[str]:
    """The controls row, in order, as what each control is."""
    row = montage.controls_row.layout()
    out = []
    for i in range(row.count()):
        widget = row.itemAt(i).widget()
        if widget is None or type(widget).__name__ == "_MatchKey":
            continue
        if isinstance(widget, ShowButton):
            out.append("Show ▾")
        elif isinstance(widget, ExportMenu):
            out.append("Export ▾")
        elif isinstance(widget, QSlider):
            out.append("slider")
        elif isinstance(widget, (QLabel, QPushButton)):
            out.append(
                widget.text().split(" ")[0]
                if widget is montage.order_button
                else widget.text()
            )
    return out


def _menu(button) -> list[str]:
    return [a.text() for a in button.menu.actions() if not a.isSeparator()]


# --- one toolbar --------------------------------------------------------------------------- #
def test_every_clip_grid_has_one_toolbar(qapp, workspace):
    grids = _grids(qapp, workspace)
    first = None
    for name, montage in grids.items():
        assert _toolbar(montage) == [
            "Most",
            "Show ▾",
            "Size",
            "slider",
            "Export ▾",
        ], name
        shape = (
            _menu(montage.show_button),
            [a.text() for a in montage.exports.button.actions_by_kind.values()],
            montage.exports.button.pick_action.text(),
        )
        first = first or shape
        assert shape == first, f"{name}'s toolbar differs from INTERPRET's"
    assert grids["AUDIT"].order_button.isHidden(), "AUDIT's order is fixed"
    assert not grids["INTERPRET"].order_button.isHidden()
    assert not grids["cluster window"].order_button.isHidden()
    context = [
        a.text() for a in grids["INTERPRET"].show_button.context_actions.values()
    ]
    assert context == ["No context", "1 s context", "2 s context", "5 s context"]


def test_the_cluster_window_has_no_permanent_show_context_or_export_rows(
    qapp, workspace
):
    window = _window(qapp, workspace._interpret, workspace.nodes["climb"])
    assert not window.findChildren(QCheckBox), "the Show row's boxes"
    assert not window.findChildren(QComboBox), "the Context box"
    texts = {b.text() for b in window.findChildren(QPushButton) if b.isVisible()}
    assert not texts & {"Clear picks", "Video…", "GIF…", "One file per clip…"}
    assert window.montage.exports.pick_bar.isHidden()


# --- Export ▾: every clip, as the grid shows them ----------------------------------------- #
def test_export_writes_every_clip_in_grid_order_with_the_show_menus_look(
    qapp, workspace, monkeypatch, tmp_path
):
    from detail.api import clip_montage
    from detail.gui import animation_export

    interpret = workspace._interpret
    montage = interpret.montage()
    montage.order_actions["atypical"].trigger()
    assert _drain(qapp, lambda: montage.order() == "atypical")
    montage.show_button.layer_actions["footage"].trigger()  # video off
    exports = montage.exports
    job = exports.job()
    grid = interpret.export_clips()
    assert len(job.plain) == len(montage.clips()) == len(interpret._order)
    assert job.plain == grid, "every clip, in the grid's order"
    assert [c.rank for c in job.plain] == list(range(1, len(grid) + 1))
    assert not job.layers.raw and job.layers.mask and not job.picks
    count = f"{len(grid):,}"
    assert exports.button.actions_by_kind["video"].text() == f"Video of {count} clips…"
    assert exports.button.describe_action.text() == (
        "Writes what the grid shows: masks + mask trail, as recorded, no context"
    )

    seen = {}

    class _Montage:
        def __init__(self, experiment_dir, clips_):
            seen["clips"] = clips_

        def export(self, path, layers_, labels_, **kw):
            seen.update(layers=layers_, labels=labels_, path=path, codec=kw["codec"])
            return path

        def close(self):
            pass

    monkeypatch.setattr(clip_montage, "ClusterMontage", _Montage)
    monkeypatch.setattr(
        animation_export,
        "run_export",
        lambda target, work, *, on_written, label: on_written(work(None)),
    )
    monkeypatch.setattr(
        animation_export, "ask_mp4_path", lambda *a: (tmp_path / "all.mp4", "mp4v")
    )
    exports.button.actions_by_kind["video"].trigger()
    assert seen["clips"] == job.clips and seen["layers"] == job.layers
    assert seen["codec"] == "mp4v", "the codec the save dialog's file type names"
    assert seen["labels"] == job.labels and seen["path"] == tmp_path / "all.mp4"
    assert "all.mp4" in exports.written.text() and exports.written.isVisibleTo(montage)


def test_the_show_menus_context_widens_the_export_and_the_tiles(qapp, workspace):
    interpret = workspace._interpret
    montage = interpret.montage()
    before = montage.exports.job()
    sources = montage.source()
    montage.show_button.context_actions[1.0].trigger()
    assert montage.context_s() == 1.0
    job = montage.exports.job()
    second = 1_000_000_000
    assert [(c.start_t_ns, c.end_t_ns) for c in job.clips] == [
        (max(c.start_t_ns - second, 0), c.end_t_ns + second) for c in before.plain
    ], "every clip widened by the context"
    assert (
        job.plain == before.plain and job.labels == before.labels
    ), "the labels and the CSV name each clip's own span"
    assert "1 s context" in montage.exports.button.describe_action.text()
    tiles = montage.source()._clips
    assert montage.source() is not sources
    assert (tiles[0].start_ns, tiles[0].end_ns) == (
        job.clips[0].start_t_ns,
        job.clips[0].end_t_ns,
    ), "the tiles play what the export writes"
    from PyQt6.QtCore import QSettings

    from detail.gui.node_workspace import CONTEXT_KEY

    assert float(QSettings().value(CONTEXT_KEY)) == 1.0, "kept for the page"


# --- pick mode -------------------------------------------------------------------------- #
def test_pick_mode_starts_with_nothing_picked_and_quit_gives_the_click_back(
    qapp, workspace
):
    interpret = workspace._interpret
    montage = interpret.montage()
    exports = montage.exports
    assert not montage.is_clickable(), "a click on INTERPRET's grid does nothing"
    exports.button.pick_action.trigger()
    assert exports.is_picking() and exports.pick_bar.isVisibleTo(montage)
    assert exports.picks() == [], "nothing picked to start (user decision, 2026-11-09)"
    assert montage.is_clickable()
    assert montage.click_hint(0).startswith("Click: pick this clip, as number 1")
    montage._clicked(3, NO_MODIFIER)
    montage._clicked(0, NO_MODIFIER)
    layer = montage.click_layer()
    assert exports.picks() == [3, 0]
    assert (layer.number(3), layer.number(0), layer.number(1)) == (1, 2, None)
    assert not layer.dropped(1), "a clip not picked is not washed out"
    assert montage.click_hint(3).startswith("Click: un-pick this clip (number 1)")
    montage._clicked(3, NO_MODIFIER)
    assert (
        exports.picks() == [0] and layer.number(0) == 1
    ), "un-picked; the rest renumber"
    assert (
        exports.pick_bar.text.text() == "Click clips to pick them, in order · 1 picked"
    )
    exports.pick_bar.quit.click()
    assert not exports.is_picking() and exports.pick_bar.isHidden()
    assert montage.click_layer() is None and not montage.is_clickable()
    assert exports.job().plain == interpret.export_clips(), "every clip again"


def test_a_shift_click_picks_the_run_from_the_last_clip_clicked(qapp, workspace):
    montage = workspace._interpret.montage()
    exports = montage.exports
    exports.start_picking()
    montage._clicked(2, NO_MODIFIER)
    montage._clicked(0, Qt.KeyboardModifier.ShiftModifier)
    assert exports.picks() == [2, 0, 1], "the run picked in grid order, onto the end"


def test_esc_leaves_pick_mode(qapp, workspace):
    montage = workspace._interpret.montage()
    montage.exports.start_picking()
    montage.canvas.setFocus()
    qapp.processEvents()
    QTest.keyClick(montage.canvas, Qt.Key.Key_Escape)
    assert not montage.exports.is_picking()


def test_esc_in_the_windows_label_field_leaves_pick_mode_not_the_window(
    qapp, workspace
):
    window = _window(qapp, workspace._interpret, workspace.nodes["climb"])
    window.montage.exports.start_picking()
    window.bar.field.edit.setFocus()
    qapp.processEvents()
    QTest.keyClick(window.bar.field.edit, Qt.Key.Key_Escape)
    assert not window.montage.exports.is_picking()
    assert window.isVisible(), "Esc left pick mode; it did not close the window"


def test_picks_export_in_pick_order_and_a_drag_drops_a_run(qapp, workspace):
    montage = workspace._interpret.montage()
    exports = montage.exports
    exports.start_picking()
    montage._clicked(2, NO_MODIFIER)
    montage._clicked(0, NO_MODIFIER)
    job = exports.job()
    assert [c.rank for c in job.plain] == [3, 1] and job.picks
    assert job.labels[0].startswith("#3 ") and job.labels[1].startswith("#1 ")
    assert exports.pick_bar.export.text() == "Export 2 ▾"
    assert exports.button.describe_action.text().endswith(", in pick order")
    # The montage's own run: a drag from a clip not picked picks the run onto the end.
    assert montage.run_between(1, 1, montage.is_off(1)) == [1]
    montage.click_layer().run([1], True)
    assert exports.picks() == [2, 0, 1]
    montage.click_layer().run([0, 1], False)
    assert exports.picks() == [2]


def test_nothing_picked_greys_the_export_with_the_reason(qapp, workspace):
    exports = workspace._interpret.montage().exports
    exports.start_picking()
    bar = exports.pick_bar.export
    assert NOTHING_PICKED == "Pick at least one clip."
    assert not bar.isEnabled() and bar.toolTip() == NOTHING_PICKED
    for action in exports.button.actions_by_kind.values():
        assert not action.isEnabled() and action.toolTip() == NOTHING_PICKED
    assert not exports.pick_bar.clear.isEnabled()
    exports.pick_bar.pick_all.click()
    n = len(workspace._interpret.montage().clips())
    assert exports.picks() == list(range(n)), "every clip, in grid order"
    assert bar.isEnabled() and bar.text() == f"Export {n} ▾"
    assert not exports.pick_bar.pick_all.isEnabled()
    exports.pick_bar.clear.click()
    assert exports.picks() == [] and not bar.isEnabled()


def test_a_double_click_opens_a_clip_without_changing_its_pick(qapp, workspace):
    """A double-click's first press is a click; the open that follows takes it back."""
    exports = workspace._interpret.montage().exports
    exports.start_picking()
    workspace._interpret.montage()._clicked(1, NO_MODIFIER)
    assert exports.picks() == [1]
    exports._on_activated(
        1
    )  # what `clip_activated` runs, here without opening the clip
    assert exports.picks() == []


def test_pick_mode_on_audit_picks_for_the_export_and_never_removes(qapp, workspace):
    audit = _audit(qapp, workspace)
    pane, montage = audit.pane, audit.pane.montage
    removals = []
    pane.take_out_requested.connect(lambda *a: removals.append(a))
    pane.put_back_requested.connect(lambda *a: removals.append(a))
    edits = dict(audit._open["edits"])
    montage.exports.start_picking()
    assert "pick" in montage.click_hint(0) and "remove" not in montage.click_hint(0)
    montage._clicked(0, NO_MODIFIER)
    montage.click_layer().run([1, 2], True)
    montage.click_layer().run([1], False)
    assert removals == [] and audit._open["edits"] == edits
    assert not audit.writes_pending()
    assert montage.exports.picks() == [0, 2]
    from detail.gui.label_clean import PICKING_HINT

    assert pane.undo_text.text() == PICKING_HINT, "the line under the top row says so"
    montage.exports.stop_picking()
    assert montage.click_hint(0).startswith("Click: remove from")
    montage._clicked(0, NO_MODIFIER)
    assert removals and removals[0][0] == [0], "a click removes again"


def test_a_reviewed_audit_cluster_can_still_be_picked_and_exported(qapp, workspace):
    audit = _audit(qapp, workspace)
    pane, montage = audit.pane, audit.pane.montage
    bar = montage.grid.verticalScrollBar()
    bar.setValue(bar.maximum())
    for _ in range(3):
        qapp.processEvents()
    pane.finish.click()
    assert _drain(qapp, lambda: not audit.writes_pending() and pane.is_finished())
    drain_busy_calls(qapp)
    assert not montage.is_clickable(), "read-only: no click removes"
    montage.exports.start_picking()
    assert montage.is_clickable()
    montage._clicked(0, NO_MODIFIER)
    assert montage.exports.picks() == [0]
    assert montage.exports.reason() is None
    montage.exports.stop_picking()
    assert not montage.is_clickable(), "read-only again"


# --- an export that finishes ends pick mode; one cancelled or failed does not ------------- #
def _picking(qapp, workspace, grid: str):
    montage = (
        workspace._interpret.montage()
        if grid == "INTERPRET"
        else _audit(qapp, workspace).pane.montage
    )
    montage.exports.start_picking()
    montage._clicked(2, NO_MODIFIER)
    montage._clicked(0, NO_MODIFIER)
    assert montage.exports.picks() == [2, 0]
    return montage


def _writer(monkeypatch, outcome: str):
    """``ClusterMontage`` stood in for: each write finishes, raises, or is cancelled — through
    the real ``busy.run_progress``, which decides what reaches the page."""
    from detail.api import clip_montage
    from detail.core.progress import OperationCancelled
    from detail.gui import busy

    errors: list[str] = []
    monkeypatch.setattr(busy, "_show_error", lambda _t, exc: errors.append(str(exc)))

    class _Montage:
        def __init__(self, experiment_dir, clips_):
            pass

        def _write(self, path):
            if outcome == "failed":
                raise RuntimeError("disk full")
            if outcome == "cancelled":
                raise OperationCancelled()
            return path

        def export(self, path, *_a, **_kw):
            return self._write(path)

        def export_clip_gifs(self, folder, *_a, **_kw):
            return [self._write(folder / "01.gif")]

        def export_clip_list(self, path, *_a, **_kw):
            return self._write(path)

        def close(self):
            pass

    monkeypatch.setattr(clip_montage, "ClusterMontage", _Montage)
    return errors


def _write(exports, kind: str, tmp_path):
    job = exports.job()
    if kind == "video":
        return exports.write_montage(job, tmp_path / "picks.mp4")
    if kind == "files":
        return exports.write_files(job, tmp_path, max_width=64, fps=5.0)
    return exports.write_list(job, tmp_path / "picks.csv")


@pytest.mark.parametrize("kind", ["video", "files", "list"])
@pytest.mark.parametrize("grid", ["INTERPRET", "AUDIT"])
def test_a_finished_export_of_the_picks_leaves_pick_mode(
    qapp, workspace, monkeypatch, tmp_path, grid, kind
):
    montage = _picking(qapp, workspace, grid)
    exports = montage.exports
    errors = _writer(monkeypatch, "finished")
    worker = _write(exports, kind, tmp_path)
    assert _drain(qapp, lambda: not exports.is_picking()), "still picking"
    assert errors == [] and exports.written.text().startswith("Wrote ")
    assert exports.picks() == [] and exports.pick_bar.isHidden()
    assert montage.click_layer() is None
    if grid == "AUDIT":
        assert montage.click_hint(0).startswith("Click: remove from"), "removes again"
    else:
        assert not montage.is_clickable()
    assert worker is not None


@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
@pytest.mark.parametrize("grid", ["INTERPRET", "AUDIT"])
def test_a_cancelled_or_failed_export_keeps_pick_mode_and_the_picks(
    qapp, workspace, monkeypatch, tmp_path, grid, outcome
):
    from detail.gui import busy

    montage = _picking(qapp, workspace, grid)
    exports = montage.exports
    errors = _writer(monkeypatch, outcome)
    _write(exports, "video", tmp_path)
    assert _drain(qapp, lambda: not busy._active), "the write never ended"
    drain_busy_calls(qapp)
    assert errors == (["disk full"] if outcome == "failed" else [])
    assert exports.is_picking() and exports.picks() == [2, 0]
    assert exports.pick_bar.isVisibleTo(montage)
    assert "pick" in montage.click_hint(1)


def test_an_export_of_every_clip_leaves_the_mode_as_it_was(
    qapp, workspace, monkeypatch, tmp_path
):
    """Not picking, an export changes nothing about the clicks: there is no mode to leave."""
    montage = workspace._interpret.montage()
    exports = montage.exports
    _writer(monkeypatch, "finished")
    exports.write_list(exports.job(), tmp_path / "all.csv")
    assert _drain(qapp, lambda: exports.written.text().startswith("Wrote "))
    assert not exports.is_picking() and montage.click_layer() is None


def test_the_pick_bars_way_out_is_quit_not_done(qapp, workspace):
    from detail.gui.montage_export import QUIT_TIP

    bar = workspace._interpret.montage().exports.pick_bar
    assert bar.quit.text() == "Quit"
    assert bar.quit.toolTip() == QUIT_TIP == "Leave pick mode without exporting (Esc)"
    assert not [b for b in bar.findChildren(QPushButton) if b.text() == "Done"]


# --- a pick session belongs to its grid and its own export (review 2026-11-09) ------------- #
def test_another_cluster_on_interpret_ends_pick_mode_and_says_so(qapp, workspace):
    """Picking survived a cluster switch: the picks of the first cluster were silently dropped
    and the next cluster's clips picked into the same export. Other clips end pick mode, the
    picks cleared, with a line saying why; the same clips re-ordered keep their picks.
    """
    interpret = workspace._interpret
    montage = _picking(qapp, workspace, "INTERPRET")
    exports = montage.exports
    interpret._on_order("random")
    assert exports.is_picking() and len(exports.picks()) == 2, "a new order keeps them"
    interpret.select(workspace.nodes["rear"])
    assert _drain(qapp, lambda: not exports.is_picking())
    assert exports.picks() == [] and exports.pick_bar.isHidden()
    assert exports.written.text() == "Pick mode ended: the grid shows other clips now."


def test_an_export_finishing_late_ends_only_its_own_pick_session(
    qapp, workspace, monkeypatch, tmp_path
):
    """An export of the picks still writing when the person quit and started picking again
    ended the new session when it finished, and threw its picks away."""
    from detail.gui import busy

    montage = _picking(qapp, workspace, "INTERPRET")
    exports = montage.exports
    _writer(monkeypatch, "finished")
    exports.write_list(exports.job(), tmp_path / "first.csv")
    exports.stop_picking()
    exports.start_picking()
    montage._clicked(1, NO_MODIFIER)
    assert _drain(qapp, lambda: not busy._active), "the write never ended"
    drain_busy_calls(qapp)
    assert exports.written.text().startswith("Wrote ")
    assert exports.is_picking() and exports.picks() == [1]
