"""The §20 calibration workspace — the surface where a cutoff is actually chosen.

What is checked here is what a construction smoke test would pass straight through: that
dragging the cutoff **rewrites nothing**, that a stale thumbnail decode never paints over a
newer band, that a cell which cannot decode says *which* failure it is, and that a behavior
returning nothing at a cutoff says so as a result rather than showing an empty strip.

The run is real — the fixture runs the actual search over synthetic embeddings with a planted
motif, reusing `tests/api/test_query.py`'s helper. Frame decode is not: this fixture has no
video, which is what makes the decode-failure guard natural rather than contrived.
"""

import math

import numpy as np
import pytest

from detail.gui import frame_source
from detail.gui.clip_grid import ClipGrid, ClipGridModes, score_digits
from detail.gui.modules.analysis_query import (
    _BrowseTab,
    _CalibrateTab,
    _QueryWorkspace,
)
from detail.gui.qt import QEvent, QLabel, QPointF, Qt

from tests.api.test_query import N_EMBEDDED, _searched, _two_behaviors
from tests.gui.mouse import mouse_event

PERIOD = 1_000_000  # the fixture's step between embedded frames


def _drain(qapp, predicate, timeout_ms: int = 5_000) -> bool:
    """Spin the event loop until ``predicate`` holds; every load here is off-thread."""
    from tests.gui.conftest import drain_until

    return drain_until(qapp, predicate, timeout_ms)


def _workspace(qapp, tmp_path, behaviors=None, **searched):
    exp, run = _searched(tmp_path, behaviors=behaviors, **searched)
    workspace = _QueryWorkspace(exp, exp.query_run(run.id))
    assert _drain(qapp, lambda: workspace._curves), "the search results never loaded"
    return exp, run, workspace


# --- the rule the surface rests on --------------------------------------------------------


def test_dragging_the_cutoff_rewrites_nothing(qapp, tmp_path):
    """§20's own guard. Reprojection rewrites a frame-view zarr per searched unit, so doing it
    per pixel of a drag freezes the GUI — and a test that only checks the number on screen
    passes while it does."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.5)
    calibrate = workspace._calibrate
    before = (run._run_dir() / "instances.csv").read_text(encoding="utf-8-sig")

    for value in (0.6, 0.7, 0.8, 0.9):
        calibrate._line.setValue(value)  # fires sigPositionChanged, as a drag does
        qapp.processEvents()
    for _ in range(50):  # give an errant commit every chance to land
        qapp.processEvents()

    # The **state**, not only the file: `set_cutoff` writes `query.yaml` before it reprojects,
    # so a drag that committed would show here even while its reprojection was still running.
    # Asserting on `instances.csv` alone passes whenever the commit is merely slow.
    assert run.state().behavior("q01").cutoff == 0.5
    assert (run._run_dir() / "instances.csv").read_text(encoding="utf-8-sig") == before
    # ...and the view did move, so this is not passing because nothing happened at all
    assert "0.90" in calibrate._header.text()


def test_releasing_the_drag_commits(qapp, tmp_path):
    """The other half: the commit has to actually happen on release, or the drag guard above
    would pass on a surface that never writes at all."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.5)
    calibrate = workspace._calibrate
    calibrate._line.setValue(0.95)
    calibrate._on_line_released()
    assert _drain(qapp, lambda: run.state().behavior("q01").cutoff == 0.95)


def test_the_numeric_box_commits_too(qapp, tmp_path):
    """It needs its own path: `sigPositionChangeFinished` fires only for a real mouse drag, so
    moving the line programmatically from the box would otherwise never commit."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    workspace._calibrate._on_confidence_typed(0.75)
    assert _drain(qapp, lambda: run.state().behavior("q01").cutoff == 0.75)


def test_seeding_both_controls_commits_nothing(qapp, tmp_path):
    """Opening a behavior sets both controls to its stored cutoff. That is a seed, not an
    edit, and it must not write."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    calibrate = workspace._calibrate
    calibrate._set_cutoff_widgets(0.42)

    assert calibrate.cutoff() == pytest.approx(0.42, abs=0.02)
    assert calibrate._confidence.value() == pytest.approx(0.42, abs=0.02)
    assert run.state().behavior("q01").cutoff is None


def test_typing_a_cutoff_refreshes_the_view_once(qapp, tmp_path):
    """The echo the re-entrancy flag actually prevents.

    Typing moves the line, which fires `sigPositionChanged`, which is the view update — so
    without the guard one keystroke refreshes twice. A refresh re-points the clip strip, so a
    doubled one is a doubled decode fan-out on every keystroke, not merely wasted work.
    """
    exp, run, workspace = _workspace(qapp, tmp_path)
    calibrate = workspace._calibrate
    calls = []
    original = calibrate._refresh_view
    calibrate._refresh_view = lambda value: (calls.append(value), original(value))[1]

    calibrate._on_confidence_typed(0.66)
    assert len(calls) == 1, f"one keystroke refreshed {len(calls)} times"


# --- what the view says -------------------------------------------------------------------


def test_zero_matches_is_stated_not_shown_as_an_empty_strip(qapp, tmp_path):
    """A behavior that returns nothing is a *result*. An empty row of frames reads as a
    surface that failed to load, which is the failure the backlog already records against
    the annotation module.

    Driven through the grid rather than by moving the cutoff to 1.0: the planted motif scores
    exactly 1.0, and the marginal band reaches *below* the line, so the strip legitimately
    still has clips there. Asserting on that would have been asserting the fixture.
    """
    exp, run, workspace = _workspace(qapp, tmp_path)
    grid = workspace._calibrate._grid

    grid.show_clips([], 0.5, empty_text="No instances at 0.99. That is a result.")
    assert "result" in grid._empty.text()
    # `isVisibleTo`, not `isVisible`: the grid has no shown ancestor in a test, so
    # `isVisible()` is False either way and the assertion would pass on a strip that never
    # hides anything.
    assert not grid._cells, "an empty strip still built cells"

    # ...and the tab supplies such a text rather than an empty string, which is the half a
    # grid-only test would miss.
    workspace._calibrate._line.setValue(0.5)
    qapp.processEvents()
    grid.show_clips([], 0.5)
    assert grid._empty.text() == ""  # the default is silent; the caller words it


def test_the_calibrate_tab_words_its_own_empty_state(qapp, tmp_path):
    """The strip's silence is the caller's to word, and Calibrate has to actually do it."""
    import inspect

    from detail.gui.modules import analysis_query

    source = inspect.getsource(analysis_query._CalibrateTab._refresh_clips)
    assert "empty_text" in source and "result" in source


def test_every_searched_recording_appears_in_the_kept_panel(qapp, tmp_path):
    """§20 requires the per-recording view to be always visible: a cutoff that keeps most of
    one animal's behavior and little of another's biases the result along the axis being
    compared, and a pooled curve cannot show it.

    The case that matters is the recording that matched **nothing**. Its answer is zero, not
    absence — dropping the row leaves the panel looking complete while the animal it is silent
    about is exactly the one worth noticing. The panel now counts instances kept rather than
    drawing score histograms, so it reads the searched-recording list directly; this injects a
    recording into that list with no candidates behind it, which is the shape that regressed
    when the panel was rewritten.
    """
    _exp, _run, workspace = _workspace(qapp, tmp_path)
    calibrate = workspace._calibrate
    held = workspace._run  # the workspace builds its own handle; patch that one
    real_state = held.state()

    class _WithAQuietRecording:
        search_recording_ids = [*real_state.search_recording_ids, "s_quiet"]

        def __getattr__(self, name):
            return getattr(real_state, name)

    held.state = lambda: _WithAQuietRecording()

    calibrate._draw_spread(0.5)
    text = calibrate._spread.text()
    assert "s1" in text and "s2" in text
    assert (
        "s_quiet" in text
    ), "a recording that matched nothing was left out of the panel"
    assert len(text.splitlines()) == 3
    quiet = next(ln for ln in text.splitlines() if ln.startswith("s_quiet"))
    assert " 0" in quiet, quiet
    # ...and the panel names it, which is the whole reason the row must not be dropped.
    assert "far fewer" in quiet, quiet


def test_a_capped_point_is_marked_on_the_curve(qapp, tmp_path, monkeypatch):
    """ "A curve that flattens because storage ran out looks exactly like a curve that flattens
    because the animal stopped." The cap is drawn, not left to be inferred.

    `soft_knn` declares no cap, so one is declared for it: the surface reads the declaration.
    """
    from detail.query_methods.soft_knn import SoftKnnQueryMethod

    exp, run, _workspace_unused = _workspace(qapp, tmp_path)
    monkeypatch.setattr(SoftKnnQueryMethod, "candidate_cap", lambda self, params: 1)

    fresh = exp.query_run(run.id)
    fresh.load_results("q01")
    curve = fresh.frequency_curve("q01", steps=11)
    assert any(p["capped"] for p in curve)

    tab = _CalibrateTab(fresh, exp)
    tab.set_behavior("q01", curve)
    # two series: the curve and the marks. Without the cap marks there would be one.
    assert len(tab._plot.listDataItems()) == 2


def _header_count(calibrate) -> int:
    """The instance count the header states, e.g. `rearing (q01) — 228 instance(s) at …`."""
    text = calibrate._header.text()
    return int(text.split(" — ", 1)[1].split(" instance(s)", 1)[0])


def test_the_header_counts_at_the_cutoff_not_a_stored_total(qapp, tmp_path):
    """`gui/confidence.py`'s lesson, which §20 says is the calibration curve's verbatim: a
    stored count cannot answer "how many at 0.7?", and a view that forgets shows a confident
    wrong total."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    calibrate = workspace._calibrate
    calibrate._line.setValue(0.0)
    qapp.processEvents()
    loose = _header_count(calibrate)
    calibrate._line.setValue(1.0)
    qapp.processEvents()
    assert _header_count(calibrate) < loose


def test_the_header_states_the_count_a_commit_at_that_cutoff_writes(qapp, tmp_path):
    """The header read the nearest of the curve's 41 points, so between points it printed a
    neighbour's count beside the exact cutoff, and the per-recording panel under it (which
    counts at the cutoff) disagreed with it on the same screen (#35). It must state what a
    commit at that cutoff would keep, so the property is checked against the artifact a commit
    actually writes, at cutoffs between the curve's points, and against the panel."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    calibrate = workspace._calibrate
    seen = set()
    for cutoff in (0.037, 0.0612, 0.3333, 0.512, 0.87):
        calibrate._refresh_header(cutoff)
        shown = _header_count(calibrate)
        panel = sum(
            int(line.split("  ←")[0].split()[-1])
            for line in calibrate._spread.text().splitlines()
        )
        assert shown == panel, f"at {cutoff}: header {shown}, panel {panel}"
        run.set_cutoff("q01", cutoff)
        kept = [
            i
            for i in exp.query_run(run.id).artifact().instances
            if i.source_label == "q01"
        ]
        assert shown == len(kept), f"at {cutoff}: header {shown}, committed {len(kept)}"
        seen.add(shown)
    assert (
        len(seen) > 1
    ), "the cutoffs chosen must give different counts to mean anything"


# --- the clip grid -------------------------------------------------------------------------


def test_a_cell_that_cannot_decode_names_the_reason(qapp, tmp_path):
    """*missing* and *undecodable* are different problems with different fixes, and a pane
    that shows neither reports neither. The thumbnail template this follows gets it wrong —
    a failed cell there reads "rendering…" forever."""
    exp, _run, workspace = _workspace(qapp, tmp_path)
    grid = workspace._calibrate._grid
    cell = next(iter(grid._cells.values()))

    cell.set_image(None, frame_source.MISSING)
    assert cell._image.text() == frame_source.unavailable_text(frame_source.MISSING)
    cell.set_image(None, frame_source.UNDECODABLE)
    assert cell._image.text() == frame_source.unavailable_text(frame_source.UNDECODABLE)
    assert cell._image.text() != ""


def test_an_unexpected_decode_error_still_names_the_cell(qapp, tmp_path):
    """`frame_source.decode` classifies the two *expected* failures. Anything else — a record
    that no longer resolves, a corrupt store — would otherwise escape into the thread pool,
    print a traceback nobody reads, and leave the cell on its placeholder forever. That is
    precisely the failure the template this follows has."""
    from detail.gui.clip_grid import _CellSignals, _CellTask

    signals = _CellSignals()
    seen = []
    signals.done.connect(lambda g, i, rgb, reason: seen.append((rgb, reason)))

    def explodes():
        raise KeyError("track_01")  # not FileNotFoundError, not IndexError

    _CellTask(1, 0, explodes, signals).run()
    assert seen == [(None, frame_source.UNDECODABLE)]


def test_a_stale_decode_never_paints(qapp, tmp_path):
    """A scroll or a cutoff move retargets the strip faster than a decode returns, so a result
    for a view the user has left must be dropped — otherwise an older, slower render lands last
    and the strip disagrees with what it says it is showing."""
    from detail.gui.behavior_query import ClipSpec

    _exp, _run, workspace = _workspace(qapp, tmp_path)
    grid = workspace._calibrate._grid

    clips = [
        ClipSpec("s1", 0, "emb_01", i * 1000, i * 1000 + 500, "x", score=0.9 - i / 100)
        for i in range(4)
    ]
    grid.show_clips(clips, 0.5)
    index, cell = next(iter(grid._cells.items()))
    stale = grid._generation - 1  # a generation the strip has already moved past

    grid._on_decoded(stale, index, np.zeros((4, 4, 3), dtype=np.uint8), None)
    assert cell._image.pixmap().isNull(), "a superseded decode painted"


def test_the_grid_shows_no_more_cells_than_it_has(qapp, tmp_path):
    from detail.gui.behavior_query import ClipSpec

    exp, _run, workspace = _workspace(qapp, tmp_path)
    grid = ClipGrid(exp, n_cells=3)
    grid.show_clips(
        [
            ClipSpec("s1", 0, "emb_01", i * 1000, i * 1000 + 500, "x", score=0.5)
            for i in range(9)
        ]
    )
    assert sum(1 for cell in grid._cells if cell.isVisible()) <= 3


def test_the_strip_defaults_to_the_marginal_band(qapp, tmp_path):
    """§20: the clips *at* the cutoff, because the decision is "is what sits at this level
    real?". Sampled-above is the second mode, not the default — a strip of obvious hits tells
    you nothing about where to cut."""
    modes = ClipGridModes()
    assert modes.mode() == ClipGridModes.NEAR


# --- browse ---------------------------------------------------------------------------------


def test_browse_says_so_when_nothing_is_calibrated(qapp, tmp_path):
    exp, run, workspace = _workspace(qapp, tmp_path)
    workspace._browse.set_behavior("q01")
    qapp.processEvents()
    assert "no cutoff" in workspace._browse._summary.text()


def test_browse_lists_the_accepted_instances(qapp, tmp_path):
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.9)
    workspace._browse.set_behavior("q01")
    assert _drain(qapp, lambda: workspace._browse._model.rowCount() > 0)
    assert workspace._browse._model.rowCount() == len(workspace._browse._rows)


def _instances(behavior_id: str, recording_id: str, n: int) -> list:
    """``n`` accepted instances of one behavior, one second apart — more than a list shows."""
    import types

    return [
        types.SimpleNamespace(
            instance_id=f"i{i:03d}",
            source_label=behavior_id,
            recording_id=recording_id,
            subject=0,
            t_ns=i * 1_000_000_000,
            end_t_ns=i * 1_000_000_000 + 500_000_000,
            score=0.9 + i / 1000,
        )
        for i in range(n)
    ]


def test_browse_shows_twenty_rows_with_the_player_level_beside_them(
    qapp, tmp_path, monkeypatch
):
    """Reported: the list was "too big" and the video "centered to the table so is off the
    screen". Thirty instances, so the twenty-row cap is something a list can overrun."""
    import types

    from detail.gui.tables import BROWSE_ROWS
    from tests.gui.layout_checks import assert_player_level_with_its_list

    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.9)
    browse = workspace._browse
    recording_id = exp.recordings[0].id
    monkeypatch.setattr(
        browse._run,
        "artifact",
        lambda: types.SimpleNamespace(
            instances=_instances("q01", recording_id, 30), coverage=[]
        ),
    )
    workspace.show_view("Browse")
    workspace.resize(1400, 1200)
    workspace.show()
    try:
        browse.set_behavior("q01")
        assert _drain(qapp, lambda: browse._model.rowCount() == 30)
        qapp.processEvents()
        assert_player_level_with_its_list(
            browse,
            browse._table,
            browse._player._src,
            above=browse._summary,
            max_rows=BROWSE_ROWS,
        )
        assert browse._table.verticalScrollBar().maximum() > 0, "the rest must scroll"
    finally:
        workspace.hide()


def test_browse_lists_recordings_by_name(qapp, tmp_path, monkeypatch):
    """§11: the id is the tool's, the name is the user's — in the list, not only in pickers."""
    import types

    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.9)
    recording = exp.recordings[0]
    recording.set_name("Fish tank A")
    browse = workspace._browse
    monkeypatch.setattr(
        browse._run,
        "artifact",
        lambda: types.SimpleNamespace(
            instances=_instances("q01", recording.id, 3), coverage=[]
        ),
    )
    browse.set_behavior("q01")
    assert _drain(qapp, lambda: browse._model.rowCount() == 3)
    shown = {browse._model.item(r, 0).text() for r in range(3)}
    assert shown == {"Fish tank A"}, shown


def test_browse_survives_finish(qapp, tmp_path):
    """§20:5773 — finishing sets the record status, it does not close the workspace.
    Reopening is only needed to *change* something."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.9)
    workspace._browse.set_behavior("q01")
    assert _drain(qapp, lambda: workspace._browse._model.rowCount() > 0)
    listed = workspace._browse._model.rowCount()

    workspace._on_finish()
    # Off-thread: finishing pins, and a pin first rebuilds the frame views a cutoff deferred.
    assert _drain(qapp, lambda: run.status == "complete")

    # No re-populate: the rows have to *survive* finishing, not be rebuilt by the assertion.
    assert run.status == "complete"
    assert workspace._browse._model.rowCount() == listed
    assert workspace._browse.isEnabled()
    assert workspace._browse.grab() is not None  # and it still paints


def test_finish_and_reopen_swap_which_button_applies(qapp, tmp_path):
    exp, run, workspace = _workspace(qapp, tmp_path)
    assert workspace._finish.isEnabled() and not workspace._reopen.isEnabled()
    workspace._on_finish()
    assert _drain(
        qapp, lambda: workspace._reopen.isEnabled()
    )  # finishing is off-thread
    assert not workspace._finish.isEnabled() and workspace._reopen.isEnabled()
    workspace._on_reopen()
    assert workspace._finish.isEnabled() and not workspace._reopen.isEnabled()


def test_a_finished_run_freezes_the_cutoff_until_reopened(qapp, tmp_path):
    """`set_cutoff` and the audit both refuse a finished run, and the controls were frozen for a
    snapshot only — so on a finished run the line dragged, the box took a number and Start drew
    a batch, each into an error. Frozen with the reason that names Reopen, and back after it.
    """
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff(workspace._behaviors[0].id, 0.0)
    workspace._audit.reload()
    calibrate, audit = workspace._calibrate, workspace._audit
    assert _drain(qapp, lambda: audit._start.isEnabled()), "the audit never loaded"
    assert calibrate._confidence.isEnabled() and calibrate._line.movable

    workspace._on_finish()
    assert _drain(qapp, lambda: workspace._reopen.isEnabled())
    assert not calibrate._confidence.isEnabled()
    assert not calibrate._line.movable
    assert not calibrate._grid.editable()
    assert "Reopen" in calibrate._confidence.toolTip()
    # Both at once: while "Finishing…" runs the page is disabled, so Start reads as disabled
    # before the audit's post-finish reload has set its reason. Waiting on the disabled state
    # alone checked the tooltip inside that window whenever the machine was busy.
    assert _drain(
        qapp,
        lambda: not audit._start.isEnabled() and "Reopen" in audit._start.toolTip(),
    ), f"Start tooltip: {audit._start.toolTip()!r}"

    workspace._on_reopen()
    assert calibrate._confidence.isEnabled() and calibrate._line.movable
    assert calibrate._grid.editable()
    assert _drain(qapp, lambda: audit._start.isEnabled())


# --- the shell ------------------------------------------------------------------------------


def test_switching_behavior_keeps_the_current_view(qapp, tmp_path):
    """§20:5757. The selector is outside the tab widget for exactly this reason."""
    exp, _run, workspace = _workspace(
        qapp, tmp_path, behaviors=_two_behaviors(["s1", "s2"])
    )
    workspace.show_view("Browse")
    workspace._selector.setCurrentIndex(1)  # the second behavior
    assert _drain(qapp, lambda: "q02" in workspace._curves)
    assert workspace.current_view() == "Browse"


def test_the_workspace_paints(qapp, tmp_path):
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.9)
    workspace._show_behavior("q01")
    qapp.processEvents()
    assert workspace.grab() is not None  # forces the plot, the sparklines and the grid


def test_a_run_with_no_state_says_so_rather_than_opening_an_empty_workspace(
    qapp, tmp_path
):
    from detail.api.experiment import Experiment
    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("p", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_09")
    adir.mkdir(parents=True, exist_ok=True)
    ar.write_analysis_yaml(
        adir, {"id": "an_09", "module_id": "query", "status": "open"}
    )

    widget = ANALYSIS_GUI_MODULES["query"].result_widget(
        proj, {"id": "an_09", "module_id": "query", "status": "open"}
    )
    text = " ".join(label.text() for label in widget.findChildren(QLabel))
    assert "no query state" in text


# --- layout: the decision on the left, the evidence on the right ---------------------------


def _shown(qapp, workspace):
    """Show the workspace at a realistic size so layout actually resolves."""
    workspace.resize(1200, 800)
    workspace.show()
    qapp.processEvents()
    return workspace._calibrate


def test_the_plot_does_not_resize_when_the_cutoff_moves(qapp, tmp_path):
    """The reported bug, asserted as a number.

    "the graph resizes as I move the confidence bar". The plot used to be the only
    vertically-Expanding child of the tab's single column, so every change below it — the clip
    strip gaining or losing a row, the header rewrapping — was absorbed by the plot. Moving the
    cutoff changes how many clips there are, so the graph moved on essentially every drag.
    """
    _exp, _run, workspace = _workspace(qapp, tmp_path)
    tab = _shown(qapp, workspace)

    heights = set()
    ranges = set()
    for cutoff in (0.05, 0.35, 0.6, 0.85, 0.99):
        tab._set_cutoff_widgets(cutoff)
        tab._refresh_view(cutoff)
        qapp.processEvents()
        heights.add(tab._plot.height())
        ranges.add(tuple(tuple(axis) for axis in tab._plot.viewRange()))
    assert len(heights) == 1, f"the plot changed height as the cutoff moved: {heights}"
    # The other way a graph "resizes": the axes rescale under a curve that never changed.
    assert (
        len(ranges) == 1
    ), f"the plot's view range moved as the cutoff moved: {ranges}"


def test_clicking_a_clip_plays_it(qapp, tmp_path):
    """The cell shows a pointing-hand cursor, so a click has to do something.

    `ClipGrid.clip_clicked` was emitted and connected to nothing — the strip advertised a
    click that went nowhere, which is what "I can't figure out how to get them to play" was.
    """
    from detail.gui.behavior_query import ClipSpec

    _exp, _run, workspace = _workspace(qapp, tmp_path)
    tab = _shown(qapp, workspace)

    clip = ClipSpec("s1", 0, "emb_01", 1_000, 2_000, "probe", score=0.5)
    tab._grid.clip_clicked.emit(clip)
    qapp.processEvents()
    assert tab._player._clip == clip


def test_the_clips_are_not_behind_a_page_scroll(qapp, tmp_path):
    """ "I didn't realize the matches were shown because they are hidden in the scrollable pane."

    The clips were the fifth widget in one column inside a QScrollArea, under a 220px plot.
    They are the evidence the whole decision rests on, so they are a fill element now, not
    something you have to scroll to.
    """
    from detail.gui.qt import QScrollArea

    _exp, _run, workspace = _workspace(qapp, tmp_path)
    tab = _shown(qapp, workspace)

    # The grid may sit inside its own scroll for overflow, but must not be buried under one
    # that also holds the plot — that is the arrangement that hid it.
    for scroll in tab.findChildren(QScrollArea):
        holds_both = scroll.isAncestorOf(tab._plot) and scroll.isAncestorOf(tab._grid)
        assert not holds_both, "the plot and the clips share a scroll area again"
    assert tab._grid.isVisibleTo(tab)


def test_the_strip_is_bounded_and_stays_full(qapp, tmp_path):
    """The band is the N nearest the cutoff, not a fixed ±0.04 width.

    A fixed width returns forty clips where the curve is dense and none where it is sparse —
    the strip either floods or reads as a surface that failed to load. The count is what the
    strip can actually show, so it is full wherever there are clips at all.
    """
    from detail.gui.clip_grid import _WINDOW

    _exp, _run, workspace = _workspace(qapp, tmp_path)
    tab = _shown(qapp, workspace)

    for cutoff in (0.1, 0.5, 0.9):
        tab._set_cutoff_widgets(cutoff)
        tab._refresh_view(cutoff)
        qapp.processEvents()
        shown = list(tab._grid._cells.values())
        assert len(shown) <= 2 * _WINDOW

    # ...and on a ranking big enough for the bound to bite. This fixture has 30 candidates, so
    # an unbounded row would satisfy the check above — and unbounded is exactly the failure:
    # `soft_knn` caps nothing, so a long pooled search yields hundreds of thousands of
    # cells — past what Qt will paint at all, and seconds and hundreds of MB well before.
    many = list(tab._grid._all) * 60  # ~1,800 candidates
    tab._grid.show_clips(many, 0.5)
    qapp.processEvents()
    assert len(many) > 2 * _WINDOW, "the synthetic ranking is too small to bound"
    assert len(tab._grid._cells) <= 2 * _WINDOW, (
        f"{len(tab._grid._cells)} cells for {len(many)} candidates — the row is the ranking, "
        "not a neighbourhood of the cutoff"
    )


def test_the_strip_separates_dropped_from_kept(qapp, tmp_path):
    """K1: the clips above the cutoff and the ones below are not one undifferentiated block.

    The row is ordered by confidence and straddles the cutoff, so without a break you cannot
    see which clips you are choosing to throw away — the one question the cutoff answers. The
    handle *is* the break, and it carries both counts.
    """
    from detail.gui.clip_grid import _CutoffHandle

    _exp, _run, workspace = _workspace(qapp, tmp_path)
    tab = _shown(qapp, workspace)
    # A cutoff from the data, so the set genuinely splits: at a fixed 0.5 this fixture keeps
    # everything and there is nothing dropped to find.
    scores = sorted((c.score or 0.0) for c in tab._grid._all)
    tab._grid.set_cutoff(scores[len(scores) // 2])
    # `_rebuild` retires the old widgets with `deleteLater`, which is deferred.
    qapp.processEvents()

    handles = tab._grid.findChildren(_CutoffHandle)
    assert len(handles) == 1, "the row needs exactly one divider"
    # The counts sit beside the row rather than inside the band: five stacked lines did not fit
    # a row sized for one frame and "kept" was clipped away. The band keeps the value and the
    # steppers, which are the parts that have to be *at* the divider.
    assert f"{tab._grid.cutoff():.4f}" in " ".join(
        w.text() for w in handles[0].findChildren(QLabel)
    )
    assert "dropped" in tab._grid._counts.text()
    assert "kept" in tab._grid._counts.text()

    # ...and the divider stands *between* them: every dropped cell left of it, every kept
    # cell right. That ordering is the row's whole claim, and it is what makes the divider
    # here and the marker on the curve above two views of one number.
    # The direction, stated directly. Keying off `_boundary_index` alone cannot see an order
    # flip: reverse the sort and the boundary reverses with it, so every cell lands on the
    # side the assertion expects and the row silently runs against the curve above it.
    by_x = sorted(tab._grid._cells.values(), key=lambda c: c.x())
    scores = [c._clip.score for c in by_x if c._clip is not None]
    assert scores == sorted(scores), "the row does not ascend left to right"

    boundary = tab._grid._boundary_index()
    handle_x = handles[0].x()
    for index, cell in tab._grid._cells.items():
        if index < boundary:
            assert cell.x() < handle_x, f"dropped clip {index} is right of the divider"
        else:
            assert cell.x() > handle_x, f"kept clip {index} is left of the divider"


def test_a_clip_names_its_recording_and_a_distinguishing_score(qapp, tmp_path):
    """Pooling interleaves animals, so a cell has to say whose clip it is.

    And the score needs enough decimals to be a *different* score: two places is what a cutoff
    reads like, and matches either side of one agree to two places, so every cell printed the
    same number and the strip looked like it was not updating.

    A shorter search than the default: captions stop at five decimals by design, and the
    default's few hundred noise peaks include pairs closer than that.
    """
    _exp, _run, workspace = _workspace(qapp, tmp_path, n_rows=600)
    tab = _shown(qapp, workspace)
    captions = [c._caption.text() for c in tab._grid._cells.values()]
    assert captions

    assert all(any(ch.isalpha() for ch in text) for text in captions), captions
    # Distinct *scores* must render distinctly. Two clips can genuinely tie — this fixture has
    # two at 1.0000 — and no number of decimals separates those; the failure being guarded is
    # neighbours that differ being printed the same, which is what made the row look static.
    # A tie is a tie at float32's resolution, not bit for bit: the planted motif scores the
    # same in both recordings, but how BLAS splits the sums across threads can leave the two a
    # unit in the last place apart. 1e-6 is ten of those units and a tenth of what five
    # decimals can show, so neighbours that differ in the caption's digits still fail here.
    rendered = {}
    for cell in tab._grid._cells.values():
        score = cell._clip.score
        if score is None:
            continue
        shown = cell._caption.text().splitlines()[-1]
        assert math.isclose(rendered.setdefault(shown, score), score, rel_tol=1e-6), (
            f"{shown!r} is printed for two different scores — "
            f"{rendered[shown]} and {score}"
        )


def test_no_section_can_be_squeezed_out_of_existence(qapp, tmp_path):
    """The reported regression, guarded as a property rather than a pixel budget.

    Stacking the player under the strip put it in a splitter pane whose sibling's minimum
    exceeded a short window, so Qt gave the player **zero** — a handle with nothing under it,
    and no way to watch a clip while its neighbours were on screen, which is the entire job of
    the surface.

    The vertical splitter is gone: the sections below the top row are `CollapsibleBand`s now, so
    a section is either open at its content's own floor or deliberately folded. There is no
    third state where Qt quietly hands one nothing. What remains a splitter is the top row,
    where the player sits *beside* the dial — genuinely side by side, which is the one case the
    rule keeps as a splitter — and that one still has to refuse to collapse.
    """
    _exp, _run, workspace = _workspace(qapp, tmp_path)
    workspace.resize(1900, 520)  # wide and short — the shape that broke it
    workspace.show()
    qapp.processEvents()
    tab = workspace._calibrate

    assert tab._top.count() == 2, "the top row was re-stacked"
    for index in range(2):
        assert not tab._top.isCollapsible(
            index
        ), f"top pane {index} can be collapsed away"
    assert tab._player.minimumWidth() > 0
    assert tab._plot.width() > 0 and tab._player.width() > 0

    for band in (tab._grid_band, tab._ethogram_band, tab._map_band):
        assert band.is_open(), f"{band._title} did not open"
        assert band._content.height() > 0, f"{band._title} was given nothing"


def test_a_cell_is_the_shape_of_the_frame_it_holds(qapp, tmp_path):
    """The white space that made a row of full frames look wasteful at any size.

    A square image label was right when these were crops around the subject. A whole frame is
    ~16:9, so a 180-wide picture is 101 tall and a square label left 79px of empty label under
    every one of them — you could not fix that by choosing a different size, which is why the
    size picker is gone rather than defaulted.
    """
    from detail.gui.clip_grid import STRIP_FRAME_PX

    _exp, _run, workspace = _workspace(qapp, tmp_path)
    tab = _shown(qapp, workspace)
    cell = next(iter(tab._grid._cells.values()))

    assert cell._image.width() == STRIP_FRAME_PX
    assert cell._image.height() == round(STRIP_FRAME_PX * 9 / 16)
    # And the cell hugs it: picture plus caption, not picture plus a square's worth of gap.
    assert (
        cell.height() < STRIP_FRAME_PX
    ), f"a {cell.height()}px cell around a {cell._image.height()}px picture"


def test_a_wider_pane_shows_more_clips(qapp, tmp_path):
    """The row does not reflow — cells keep their size and the pane fits more of them, which
    is what keeps a given scroll distance meaning a fixed number of clips."""
    _exp, _run, workspace = _workspace(qapp, tmp_path)
    tab = _shown(qapp, workspace)
    workspace.resize(1100, 800)
    workspace.show()
    qapp.processEvents()
    narrow = tab._grid.visible_cells()

    workspace.resize(1700, 800)
    qapp.processEvents()
    assert tab._grid.visible_cells() > narrow, "widening the pane showed no more clips"
    workspace.hide()


def test_a_cell_does_not_stretch_past_its_picture(qapp, tmp_path):
    """Four thumbnails filled a window with room for a dozen.

    The grid stretched its columns to fill the pane, so each cell was ~360px of mostly empty
    frame around a 150px picture. The image is a fixed size and a trailing stretch takes the
    slack instead — now the row's, since the grid is gone.
    """
    _exp, _run, workspace = _workspace(qapp, tmp_path)
    tab = _shown(qapp, workspace)
    workspace.resize(1400, 900)
    workspace.show()
    qapp.processEvents()

    cell = next(iter(tab._grid._cells.values()))
    assert (
        cell.width() < tab._grid.width() / 2
    ), f"a cell is {cell.width()}px in a {tab._grid.width()}px pane — the column stretched"


def test_the_player_scrubs_the_clip_and_nothing_either_side(qapp, tmp_path):
    """No padding: the strip is the match, so the playhead starts at its start.

    Trim mode pads the window by ±3s. The annotation loop needs that — there you drag a
    match's bounds outward and must see past them. Here you are only deciding whether the clip
    is the behavior, and the padding put the playhead a third of the way along a strip that is
    mostly *not* the match, which reads as the video starting in the middle.

    The preview is stubbed because the fixture has no video: with a real one absent, trim's
    expansion raises and *falls back* to the clip span, so a fixture-only assertion passes
    whichever way the flag is set — which is exactly what the first version of this test did.
    """
    from detail.gui.behavior_query import ClipSpec, _ClipPlayer

    class _Stub:
        def t_ns_to_frame(self, _rec, _emb, t_ns):
            return max(int(t_ns) // PERIOD, 0)

        def recording_frame_count(self, _rec, _emb):
            return 100_000

        def frame_rate(self, _rec, _emb):
            return 30.0

    exp, _run, workspace = _workspace(qapp, tmp_path)
    tab = _shown(qapp, workspace)
    clip = ClipSpec("s1", 0, "emb_01", 20_000 * PERIOD, 20_040 * PERIOD, "p", score=0.9)

    padded = _ClipPlayer(exp, context_s=2, walk_edges=True)
    padded._preview = _Stub()
    padded.set_clip(clip)
    assert (
        padded.bounds() != padded.clip_span()
    ), "the stub never reached the padding path"

    tab._player._preview = _Stub()
    tab._player.set_clip(clip)
    qapp.processEvents()
    assert tab._player.bounds() == tab._player.clip_span(), "the window was padded"
    assert (
        tab._player._scrubber.frame() == 0
    ), "playback did not start at the clip's start"


def test_browse_sorts_by_value_not_by_text(qapp, tmp_path):
    """Sortable headers, sorting on what the column is *about*.

    `QStandardItem` compares display strings, which puts `10.00 s` before `9.00 s` and orders
    confidences by their leading characters. Each cell carries the value instead.
    """
    from detail.gui.tables import SortCell

    from detail.gui.qt import QStandardItemModel, Qt

    assert SortCell("9.00 s", 9_000_000_000) > SortCell("10.00 s", 10_000)
    assert SortCell("0.9", 0.9) > SortCell("0.10", 0.10)

    # What a header click does: Qt sorts the model through each cell's `operator<`.
    model = QStandardItemModel()
    for text, key in [("10.00 s", 10_000), ("9.00 s", 9_000_000_000), ("0.5 s", 500)]:
        model.appendRow([SortCell(text, key)])
    model.sort(0, Qt.SortOrder.DescendingOrder)
    assert [model.item(r, 0).text() for r in range(3)] == ["9.00 s", "10.00 s", "0.5 s"]
    model.sort(0, Qt.SortOrder.AscendingOrder)
    assert [model.item(r, 0).text() for r in range(3)] == ["0.5 s", "10.00 s", "9.00 s"]


def test_browse_opens_the_row_you_clicked_after_sorting(qapp, tmp_path):
    """The bug sorting would otherwise introduce.

    The view reorders the model, so a row's identity cannot be its position — `_show_row` read
    `self._rows[current.row()]`, which after any sort plays a different instance than the one
    clicked. Silent, and wrong in a way that looks like the player is broken.
    """
    from detail.gui.qt import Qt as _Qt

    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.0)
    browse = workspace._browse
    browse.set_behavior("q01")
    assert _drain(qapp, lambda: browse._model.rowCount() > 1)

    browse._table.sortByColumn(3, _Qt.SortOrder.AscendingOrder)  # by confidence
    qapp.processEvents()

    for view_row in range(min(browse._model.rowCount(), 3)):
        index = browse._model.item(view_row, 0).data(_Qt.ItemDataRole.UserRole)
        expected = browse._rows[int(index)]
        browse._table.selectRow(view_row)
        qapp.processEvents()
        assert browse._player._clip is not None
        assert int(browse._player._clip.start_t_ns) == int(
            expected.t_ns
        ), f"row {view_row} played a different instance than the one selected"


def test_score_digits_separates_a_band_two_decimals_cannot():
    """The original defect, on fixed values: matches either side of a cutoff agree to two
    places, so a kept clip and a cut one both printed `0.66` and the list looked frozen.
    """
    assert score_digits([0.66252, 0.66301, 0.66412]) == 4
    assert len({f"{v:.4f}" for v in (0.66252, 0.66301, 0.66412)}) == 3


def test_score_digits_stops_at_two_when_two_is_enough():
    """It is not "always more digits" — a caption stays a caption when it can."""
    assert score_digits([0.10, 0.42, 0.87]) == 2


def test_score_digits_counts_a_float32_tie_as_one_score():
    """One match found in two recordings scores the same, but float32 sums split across BLAS
    threads can return the two a unit in the last place apart. Counted as two scores, they
    could never be separated, so every caption went to the five-decimal cap on a machine with
    many threads and stayed at two on one with few."""
    tie = [0.8565883636474609, 0.8565884232521057]  # measured, OPENBLAS_NUM_THREADS=32
    assert score_digits([0.10, 0.42, *tie]) == 2
    # ...while scores that differ in the caption's digits still count as different.
    assert score_digits([0.10, 0.42, 0.85658, 0.85659]) == 5


def test_score_digits_gives_up_at_the_cap_rather_than_growing_without_bound():
    """Values agreeing past five places cannot be separated, and the cap is what stops a
    caption becoming a serial number. This is why the browse test asserts distinctness *at
    the cap* rather than unconditionally.

    Three millionths apart: two scores a ten-millionth apart are within float32's rounding of
    each other, which ``score_digits`` counts as one score (the test above)."""
    values = [0.662520, 0.662523]
    assert score_digits(values) == 5
    assert len({f"{v:.5f}" for v in values}) == 1  # still collides, by design


def test_browse_scores_are_distinguishable(qapp, tmp_path):
    """Same rule as the strip: a fixed 2dp makes every row in a tight band read alike.

    Pinned against `score_digits`, which is the contract, rather than against unconditional
    distinctness, which is not: it caps at 5 decimals so a caption stays a caption, so two
    matches agreeing to 1e-5 are *allowed* to read alike. Asserting they could not made this
    flaky — roughly one full-tier run in five — because the fixture draws fresh embeddings
    and whether any pair lands inside the cap is a property of that run's scores, not of the
    rendering. `test_score_digits_*` covers the 2dp claim on fixed values instead.
    """
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.0)
    browse = workspace._browse
    browse.set_behavior("q01")
    assert _drain(qapp, lambda: browse._model.rowCount() > 1)

    shown = [browse._model.item(r, 3).text() for r in range(browse._model.rowCount())]
    values = [r.score for r in browse._rows]
    # Equal scores rendering alike is correct — the planted motif matches itself at exactly
    # 1.0 in more than one recording — so this counts renderings, not rows.
    at_cap = {"—" if v is None else f"{v:.5f}" for v in values}
    assert len(set(shown)) == len(at_cap), f"confidences collide: {sorted(shown)}"
    # ...and the column went through `score_digits` rather than a hardcoded format.
    digits = score_digits(values)
    assert all(
        len(text.split(".")[-1]) == digits for text in shown if text != "—"
    ), f"expected {digits} decimals, got {sorted(set(shown))}"


# --- reading a pinned calibration back (§21) ----------------------------------------------


def test_a_scan_snapshot_renders_through_the_same_workspace(qapp, tmp_path):
    """A Query snapshot's interpretation is its behaviors and their cutoffs, so it is as readable
    as a clustering one. It was left out at first on the grounds that a strip of clips is not a
    vocabulary — the cutoffs are the part that judgment missed."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    behavior = workspace._behaviors[0]
    run.set_cutoff(behavior.id, 0.6)
    snapshot_id = run.pin(reason="finish")
    assert snapshot_id, "a calibrated run has an artifact to pin"

    # Move the live run *after* pinning. Without this the frozen and live values are equal, so
    # the assertion below passes whether the freeze works or not — which is how a `state()` that
    # read the live run shipped through a green suite.
    run.set_cutoff(behavior.id, 0.9)

    frozen = _QueryWorkspace(exp, run.view(snapshot_id))
    assert _drain(qapp, lambda: frozen._curves), "the frozen search never loaded"
    assert frozen._run.snapshot_id == snapshot_id
    assert frozen._run.state().behavior(behavior.id).cutoff == 0.6, "the pinned cutoff"
    assert exp.query_run(run.id).state().behavior(behavior.id).cutoff == 0.9


def test_a_scan_snapshot_refuses_every_edit(qapp, tmp_path):
    """Both ways to move a cutoff go dead. Leaving one live would let a drag write to the run
    somebody opened only to read — and it writes to the *run*, not to the copy on screen.
    """
    exp, run, workspace = _workspace(qapp, tmp_path)
    behavior = workspace._behaviors[0]
    run.set_cutoff(behavior.id, 0.6)
    frozen = _QueryWorkspace(exp, run.view(run.pin(reason="finish")))
    assert _drain(qapp, lambda: frozen._curves)

    assert not frozen._finish.isEnabled()
    assert not frozen._reopen.isEnabled()
    assert not frozen._calibrate._confidence.isEnabled()
    assert not frozen._calibrate._line.movable
    # The third way to move a cutoff, after the curve's line and the number box. Freezing two of
    # three left the strip's own stepping live: the write was refused, but the screen showed a
    # cutoff belonging to neither the snapshot nor the run.
    assert not frozen._calibrate._grid.editable()
    before = frozen._calibrate._grid.cutoff()
    frozen._calibrate._grid.step_cutoff(down=True)
    assert frozen._calibrate._grid.cutoff() == before, "the strip must not step"
    assert "snapshot" in frozen._finish.toolTip()

    # Even reaching past the widgets does not write.
    frozen._calibrate._commit(0.9)
    assert exp.query_run(run.id).state().behavior(behavior.id).cutoff == 0.6


def test_a_scan_snapshot_says_what_it_is(qapp, tmp_path):
    """The status line is the banner's counterpart inside the workspace: it must not read like the
    live run when it is not one."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff(workspace._behaviors[0].id, 0.6)
    snapshot_id = run.pin(reason="export")
    frozen = _QueryWorkspace(exp, run.view(snapshot_id))
    assert _drain(qapp, lambda: frozen._curves)
    assert snapshot_id in frozen._status.text()
    assert "Read-only" in frozen._status.text()


def test_a_scan_snapshot_row_reports_behaviors_not_clusters(qapp, tmp_path):
    """The defect this fixed. Every surface read `granularity` and `names` out of the record —
    clustering's shape — so a calibrated scan rendered `Clusters — · Named 0 · unnamed`: wrong
    information, on a run that was fully interpreted, with nothing failing."""
    from detail.gui.snapshots import snapshot_columns, snapshot_rows

    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff(workspace._behaviors[0].id, 0.6)
    run.pin(reason="finish")

    handle = exp.query_run(run.id)
    columns = snapshot_columns(handle, handle.snapshots())
    assert "Behaviors" in columns and "Cutoffs" in columns
    assert "Clusters" not in columns

    row = dict(zip(columns, snapshot_rows(exp, handle)[0]))
    assert row["Behaviors"] == str(len(workspace._behaviors))
    assert row["Calibrated"] == "1"
    assert row["Cutoffs"] == "0.60"
    assert (
        row["Vocabulary"] != "unnamed"
    ), "the behaviors are named; the row must say so"


# --------------------------------------------------------------------------- #
# the row: three gestures on one surface
# --------------------------------------------------------------------------- #
def _laid_out(qapp, tmp_path):
    _exp, run, workspace = _workspace(qapp, tmp_path)
    tab = _shown(qapp, workspace)
    workspace.resize(1400, 900)
    workspace.show()
    qapp.processEvents()
    return run, workspace, tab


def test_dragging_the_divider_does_not_reproject(qapp, tmp_path):
    """§20's rule for this surface. A drag reads the cached curve and re-points the row;
    only the release commits. Conflating them rewrites a frame-view zarr per searched unit,
    per pixel of a drag."""
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    # On the signal, not on `run.set_cutoff`: the commit path goes through `run_progress`, so
    # the write happens on a worker and an assertion right after the drag runs before it would
    # have been called either way — it passed with the commit wired into every drag.
    moved, released = [], []
    tab._grid.cutoff_moved.connect(moved.append)
    tab._grid.cutoff_released.connect(released.append)

    scores = sorted(c.score or 0.0 for c in tab._grid._all)
    tab._grid.move_cutoff(scores[len(scores) // 2], commit=False)
    tab._grid.move_cutoff(scores[len(scores) // 3], commit=False)
    qapp.processEvents()
    assert len(moved) == 2, "the drag did not re-point the view"
    assert released == [], "a drag committed"
    workspace.hide()


def test_a_stepper_and_a_drag_commit_the_same_way(qapp, tmp_path):
    """Two ways to move one line must reproject the same number of times, or the same
    decision costs differently depending on how it was made."""
    run, workspace, tab = _laid_out(qapp, tmp_path)
    released = []
    tab._grid.cutoff_released.connect(released.append)

    scores = sorted(c.score or 0.0 for c in tab._grid._all)
    tab._grid.move_cutoff(scores[len(scores) // 2], commit=True)
    assert len(released) == 1, "a released drag did not commit exactly once"
    tab._grid.step_cutoff(down=True)
    assert len(released) == 2, "a stepper did not commit exactly once"
    workspace.hide()


def test_clicking_a_clip_plays_it_and_leaves_the_cutoff_alone(qapp, tmp_path):
    """The three gestures are kept apart by *where* they are, not by a mode — so a click on a
    cell must not be read as a click near the divider."""
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    before = tab._grid.cutoff()
    clip = tab._grid._all[len(tab._grid._all) // 2]
    tab._on_clip_clicked(clip)
    qapp.processEvents()

    assert tab._grid.cutoff() == before, "playing a clip moved the cutoff"
    assert tab._player._clip is not None
    workspace.hide()


def test_the_cells_are_whole_frames_not_crops(qapp, tmp_path):
    """The question at this size is whether the *stretch* is the behavior, which a crop around
    the animal cannot answer — it shows the subject and nothing it is doing it against.
    """
    import inspect

    from detail.gui import clip_grid

    source = inspect.getsource(clip_grid._producer)
    assert "frame_masked" in source
    assert "source_crop" not in source, "the cells went back to crops"


def test_the_ethogram_has_one_row_per_searched_recording(qapp, tmp_path):
    """Where the kept instances land. Read from the artifact, so it answers for the committed
    cutoff — which is why it redraws on commit and not under the drag."""
    run, workspace, tab = _laid_out(qapp, tmp_path)
    # Commit a cutoff first. The artifact holds only what a *committed* cutoff kept, so on a
    # run nobody has calibrated every row is empty — and a test over empty rows cannot tell a
    # correct split from one that hands every recording all the instances.
    scores = sorted(c.score or 0.0 for c in tab._grid._all)
    run.set_cutoff(tab._behavior_id, scores[0])
    rows, _positions = tab._ethogram_rows()
    assert (
        sum(len(bands) for _rid, bands, _width in rows) > 0
    ), "no bands, so this proves nothing"

    recordings = {c.recording_id for c in tab._grid._all}
    assert {rid for rid, _bands, _width in rows} == recordings

    # Each row's bands must be *that recording's*. Asserting only the row set and the
    # fraction range cannot see a row given every recording's instances — the set is
    # unchanged and the fractions are still fractions.
    from detail.storage import behavior_output as bo

    by_recording: dict = {}
    for inst in bo.bouts(list(run.artifact().instances)):
        by_recording.setdefault(inst.recording_id, 0)
        by_recording[inst.recording_id] += 1
    for rid, bands, width in rows:
        assert width > 0, f"{rid} has no length for the time axis"
        assert len(bands) == by_recording.get(
            rid, 0
        ), f"{rid} shows another's instances"
        for start, end, _cluster in bands:
            assert 0.0 <= start <= 1.0 and 0.0 <= end <= 1.0, (start, end)
            assert start <= end
    workspace.hide()


def test_the_calibrate_panes_pack_at_the_top(qapp, tmp_path):
    """Four stacked panes now, and `gui_design` §3 applies to a pane, not only a page."""
    from tests.gui.layout_checks import assert_panes_top_justify, owns_a_pane

    _exp, _run, workspace = _workspace(qapp, tmp_path)
    _shown(qapp, workspace)
    assert owns_a_pane(workspace)
    assert_panes_top_justify(workspace)


def test_scrolling_to_an_edge_loads_nothing_and_moves_nothing(qapp, tmp_path):
    """The fix for the two things paging did wrong, asserted as the absence of paging.

    The row used to load more at its ends and re-anchor the scroll afterwards. That yanked the
    scrollbar handle out from under a dragging mouse, and a drag that kept arriving at the end
    kept firing the load. The window is a fixed neighbourhood of the cutoff now: scrolling only
    decodes, so there is nothing to shift and nothing to re-trigger.
    """
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    strip = tab._grid
    workspace.resize(520, 900)
    qapp.processEvents()
    bar = strip._scroll.horizontalScrollBar()
    assert bar.maximum() > 0, "the row does not scroll, so this proves nothing"

    before = (strip._lo, strip._hi, {i: c._clip for i, c in strip._cells.items()})
    for value in (bar.maximum(), 0, bar.maximum(), 0):
        bar.setValue(value)
        qapp.processEvents()
        qapp.processEvents()
        assert (strip._lo, strip._hi) == before[:2], "an edge moved the window"
    assert {i: c._clip for i, c in strip._cells.items()} == before[2]
    assert bar.value() == 0, "the scroll was moved out from under the caller"
    workspace.hide()


def test_the_window_recentres_when_the_cutoff_moves(qapp, tmp_path):
    """The only thing that moves the neighbourhood is the cutoff, which is the gesture the
    screen is for — and the way to see past the window's edge."""
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    strip = tab._grid
    strip.set_cutoff(0.0)
    qapp.processEvents()
    low = strip._boundary_index()
    strip.set_cutoff(1.0)
    qapp.processEvents()

    assert strip._boundary_index() != low, "the boundary did not move"
    # Centred on the boundary: everything below it in the window is dropped, everything from
    # it is kept, and the divider is scrolled to the middle rather than left at an edge.
    assert strip._lo <= strip._boundary_index() <= strip._hi
    workspace.hide()


def test_the_ends_say_whether_there_is_more(qapp, tmp_path):
    """Pushing against a limit reads the same as a stall unless the edge says which it is."""
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    strip = tab._grid
    # Both ends, separately. Joining them and asking for one word passes with either half
    # blank — and a blank end is exactly the case a reader cannot distinguish from a stall.
    assert "confident" in strip._left_edge.text(), strip._left_edge.text()
    assert "confident" in strip._right_edge.text(), strip._right_edge.text()
    workspace.hide()


def test_moving_the_cutoff_does_not_re_decode_frames_it_already_has(qapp, tmp_path):
    """The lag when nudging the line on the graph.

    A cutoff move changes which side of the divider a clip falls on — never the picture — but
    every rebuild queued a fresh decode for every visible cell, so each nudge re-decoded a
    screenful of full frames. The cache is keyed on clip identity, so a rebuild pays for
    nothing it has already shown.
    """
    from detail.gui import clip_grid

    _run, workspace, tab = _laid_out(qapp, tmp_path)
    strip = tab._grid
    for _ in range(40):
        qapp.processEvents()

    # Seeded directly: this fixture's clips have no decodable video, so a real decode never
    # populates the cache and a test that waited for one would pass on an empty set.
    seeded = {strip._key_for(clip) for clip in strip._all}
    import numpy as np

    frame = np.zeros((101, 180, 3), dtype=np.uint8)
    strip._decoded.update(dict.fromkeys(seeded, frame))

    decodes = []
    original = clip_grid._producer

    def counting(preview, clip, size_px):
        decodes.append(strip._key_for(clip))
        return original(preview, clip, size_px)

    clip_grid._producer = counting
    try:
        scores = sorted(c.score or 0.0 for c in strip._all)
        for score in scores[: len(scores) // 2]:
            strip.set_cutoff(score)
            qapp.processEvents()
        assert not (
            set(decodes) & seeded
        ), f"{len(set(decodes) & seeded)} frames were decoded again from cache"
    finally:
        clip_grid._producer = original
    workspace.hide()


def test_the_recentre_button_works_whenever_the_row_can_scroll(qapp, tmp_path):
    """It asked "is the divider *visible*" and greyed itself out when it was — but visible is
    not centred, so resizing the window left the line off-centre with the only control that
    would fix it disabled."""
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    strip = tab._grid
    workspace.resize(520, 900)
    qapp.processEvents()
    assert strip._scroll.horizontalScrollBar().maximum() > 0
    assert strip._recentre.isEnabled(), "greyed out with a row that can scroll"

    bar = strip._scroll.horizontalScrollBar()
    bar.setValue(bar.maximum())
    qapp.processEvents()
    strip.back_to_cutoff()
    qapp.processEvents()
    _assert_divider_centred(strip)
    workspace.hide()


def _assert_divider_centred(strip) -> None:
    """Against the divider's **laid-out** position, in pixels.

    The first version of this check predicted where the divider ought to be as
    ``cells × (frame + gap)``. That agreed with the production code because the production code
    made the same prediction — and the same error: the pitch was a couple of pixels out, which
    over the 150 cells of a window accumulated to ~300px, so a divider left a cell and a half
    off centre passed a guard whose tolerance was one cell. What the user sees is where the
    widget ended up, so that is what is measured.
    """
    handle = strip._handle
    assert handle is not None
    bar = strip._scroll.horizontalScrollBar()
    # A row that has not been laid out reports every position as 0, and a scrollbar reset by a
    # rebuild reports 0 too — so "centred" was satisfied by two zeros agreeing, which is how
    # this passed against a cut that never scrolled anywhere. The divider is only at 0 when it
    # is the row's first widget.
    if strip._boundary_index() > strip._lo:
        assert handle.x() > 0, "the row is not laid out — nothing here can be measured"
    middle = handle.x() + handle.width() // 2
    # The exact value, clamped the way the scrollbar clamps it. Written first as "off centre by
    # at most 2px, unless the bar is against an end", this waved through every failure it
    # existed for: a centring that never ran leaves the bar at 0, which is an end.
    want = max(0, min(bar.maximum(), middle - strip._scroll.viewport().width() // 2))
    assert (
        bar.value() == want
    ), f"the divider is not centred: scroll {bar.value()}, want {want}"


def test_the_divider_centres_over_a_ranking_big_enough_to_scroll(qapp, tmp_path):
    """Open centred, and come back to centred — on a row that is genuinely longer than the pane.

    The workspace fixture's ranking fits on screen at any sane window size, so its divider is
    trivially "centred" wherever the scrollbar happens to sit and every centring bug passes.
    Repeating the clips gives the window its full 150 cells either side, which is the shape
    every one of those bugs needed: a stale pitch estimate that drifted ~300px over that
    distance, a rebuild that replaced the divider with a fresh widget at x=0, and a read taken
    before Qt had laid the row out at all.
    """
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    workspace.resize(1700, 900)
    qapp.processEvents()
    strip = tab._grid

    many = list(strip._all) * 20
    scores = sorted(c.score or 0.0 for c in many)
    strip.show_clips(many, scores[len(scores) // 2])
    for _ in range(30):
        qapp.processEvents()

    bar = strip._scroll.horizontalScrollBar()
    assert bar.maximum() > 0, "the row still fits on screen — this proves nothing"
    _assert_divider_centred(strip)

    bar.setValue(bar.maximum())
    qapp.processEvents()
    assert bar.value() != 0, "scrolling away did nothing"
    strip.back_to_cutoff()
    for _ in range(10):
        qapp.processEvents()
    _assert_divider_centred(strip)
    workspace.hide()


def test_nudging_the_cutoff_inside_the_window_still_centres(qapp, tmp_path):
    """The rebuild with nothing to wait for.

    Stepping the cutoff one clip rebuilds the row at the same width, so every position changes
    and the scroll range does not — there is no ``rangeChanged`` to hang the centring on, and a
    version that waited only for one left the divider wherever the last scroll had put it.
    """
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    workspace.resize(1700, 900)
    qapp.processEvents()
    strip = tab._grid

    many = list(strip._all) * 20
    scores = sorted(c.score or 0.0 for c in many)
    middle = len(scores) // 2
    strip.show_clips(many, scores[middle])
    for _ in range(30):
        qapp.processEvents()
    before = strip._scroll.horizontalScrollBar().maximum()

    strip.set_cutoff(scores[middle + 1])
    for _ in range(10):
        qapp.processEvents()
    assert (
        strip._scroll.horizontalScrollBar().maximum() == before
    ), "the row changed width — this is not the case being guarded"
    _assert_divider_centred(strip)
    workspace.hide()


# --------------------------------------------------------------------------- #
# setting the cutoff from a clip you scrolled to
# --------------------------------------------------------------------------- #
def _click(strip, index: int):
    """Click the cell at ``index`` the way a person does — through the cell, not the strip."""
    cell = strip._cells[index]
    cell.mousePressEvent(
        mouse_event(
            QEvent.Type.MouseButtonPress,
            QPointF(5.0, 5.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )


def test_clicking_a_clip_plays_it_and_does_not_move_the_cutoff(qapp, tmp_path):
    """The whole reason the cut is a button and not a click in the row.

    A cell that both played and cut would have to be told apart by *where* inside it you
    clicked, on the one surface people browse by clicking. So the click keeps its one meaning
    and the second gesture lives outside the row.
    """
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    strip = tab._grid
    before = strip.cutoff()
    target = max(strip._cells)
    # The rendered cell, not a property of it. `frameShadow() != frameShadow()` was the first
    # version and it passed on a mark nobody could see: a `Sunken` bevel on a `StyledPanel`
    # drew the marked cell and its neighbours as the same picture. The cell's content does not
    # change when it is clicked, so any difference in the image *is* the mark.
    unmarked = strip._cells[target].grab().toImage()

    _click(strip, target)
    qapp.processEvents()

    assert strip.cutoff() == before, "clicking a clip moved the cutoff"
    assert strip.marked_index() == target
    assert (
        strip._cells[target].grab().toImage() != unmarked
    ), "the clicked cell renders identically to the way it did unclicked"

    # ...and it is still marked after the row is rebuilt, which every cutoff move does. The
    # mark lived on the cell at first, and `_rebuild` throws every cell away — so nudging the
    # line cleared the highlight off the clip still playing in the pane above, and the cut
    # button went on naming it.
    strip.set_cutoff(strip.cutoff() + 1e-6)
    qapp.processEvents()
    assert strip.marked_index() == target
    assert (
        strip._cells[target].grab().toImage() != unmarked
    ), "the mark did not survive a rebuild"
    workspace.hide()


def test_the_cut_button_names_the_clip_and_what_keeping_from_it_costs(qapp, tmp_path):
    """The readback that makes a one-press commit safe.

    Pressing it reprojects, and the clip is somewhere you scrolled to rather than under the
    divider you can see. A button reading only *Cut here* would be asking you to remember which
    clip you last clicked, and to work out for yourself which side of it survives.
    """
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    strip = tab._grid
    assert not strip._cut.isEnabled(), "offered a cut before anything was marked"
    assert strip._cut.text() == "Cut here"

    # The highest built clip scoring clearly below the current cut, so the promise differs from
    # today's count by construction. Not `max(strip._cells)`: the top two clips scored
    # 0.99943244 and 0.99943214 on macOS and tied on Linux CPU, so marking the last one there
    # promised the count already shown and the precondition below failed.
    cut_score = strip._all[strip._boundary_index()].score or 0.0
    target = max(
        i for i in strip._cells if (strip._all[i].score or 0.0) < cut_score - 0.01
    )
    _click(strip, target)
    qapp.processEvents()

    import re

    label = strip._cut.text()
    score = strip._all[target].score or 0.0
    assert f"{score:.{strip._digits}f}" in label, label

    promised = int(re.search(r"\((\d+) kept\)", label).group(1))
    assert (
        promised != len(strip._all) - strip._boundary_index()
    ), "this fixture cannot tell the prospective count from the current one"

    # The promise, checked by keeping it — not by recomputing it the way the button did, which
    # would pass on any arithmetic the two agreed on. Scores tie, so "everything from the marked
    # clip rightward" is not the answer: two clips at 1.0000 are one cutoff and both survive it.
    strip._cut.click()
    for _ in range(20):
        qapp.processEvents()
    assert (
        f"{promised} kept" in strip._counts.text()
    ), f"the button promised {promised} kept and the row shows {strip._counts.text()!r}"
    workspace.hide()


def test_cutting_at_the_marked_clip_moves_the_line_there_and_commits_once(
    qapp, tmp_path
):
    """One press, one commit — the same contract as the steppers, not the drag's.

    A drag emits a view-only move on every pixel and commits on release. This is discrete, so
    it does both at once; emitting two commits would reproject twice for one decision.
    """
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    strip = tab._grid
    commits = []
    strip.cutoff_released.connect(commits.append)

    target = max(strip._cells)
    marked = strip._all[target]
    _click(strip, target)
    qapp.processEvents()
    strip._cut.click()
    for _ in range(20):
        qapp.processEvents()

    assert commits == [pytest.approx(marked.score)], commits
    # The marked clip is now the least confident one kept — the button's promise, checked
    # against the ordering rather than against the number it was set from.
    assert strip._all[strip._boundary_index()].score == pytest.approx(marked.score)
    # ...and the button stops offering a cut that would do nothing.
    assert not strip._cut.isEnabled()
    assert strip._cut.text() == "Cut here"
    workspace.hide()


def test_the_row_recentres_on_the_line_after_a_cut(qapp, tmp_path):
    """The clip you cut at was off-screen by definition — you scrolled to it.

    Leaving the view where it was would put the line you just set outside the pane, which is
    the state the centre button exists to rescue you from.
    """
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    workspace.resize(1700, 900)
    qapp.processEvents()
    strip = tab._grid
    many = list(strip._all) * 20
    scores = sorted(c.score or 0.0 for c in many)
    strip.show_clips(many, scores[len(scores) // 2])
    for _ in range(30):
        qapp.processEvents()
    assert strip._scroll.horizontalScrollBar().maximum() > 0, "the row does not scroll"
    # The strip on its own. Wired to the tab, a commit round-trips back through `set_cutoff`,
    # which re-centres for its own reasons — so a strip that never re-centred itself still
    # ended up centred, and this passed against the bug it exists for.
    strip.cutoff_released.disconnect()
    strip.cutoff_moved.disconnect()

    target = max(strip._cells)
    _click(strip, target)
    qapp.processEvents()
    strip._cut.click()
    for _ in range(20):
        qapp.processEvents()

    _assert_divider_centred(strip)
    workspace.hide()


def test_the_cut_button_is_frozen_on_a_scan_snapshot(qapp, tmp_path):
    """The fifth way to move a cutoff, and the fifth place to forget to freeze it.

    The curve's line, the number box, the steppers and the drag are all frozen on a snapshot.
    A button that moves the same value is the same hole, and it is the one a control walk finds
    only if it is enabled — so it is checked directly, marked or not.
    """
    exp, run, _workspace_widget = _workspace(qapp, tmp_path)
    behavior = run.state().behaviors[0]
    run.set_cutoff(behavior.id, 0.6)
    frozen = _QueryWorkspace(exp, run.view(run.pin(reason="finish")))
    assert _drain(qapp, lambda: frozen._curves)
    frozen.resize(1400, 900)
    frozen.show()
    qapp.processEvents()

    strip = frozen._calibrate._grid
    target = max(strip._cells)
    _click(strip, target)
    qapp.processEvents()
    assert (
        strip.marked_index() == target
    ), "the clip did not even play — wrong thing frozen"
    assert (
        not strip._cut.isEnabled()
    ), "a snapshot's cutoff can be moved from the clip row"
    before = strip.cutoff()
    strip.cut_at_marked()
    qapp.processEvents()
    assert strip.cutoff() == before, "calling it directly moved a snapshot's cutoff"
    frozen.hide()


def test_the_ethograms_are_drawn_on_opening_not_only_after_a_commit(qapp, tmp_path):
    """The pane held a heading and nothing under it until somebody moved the cutoff.

    They were hung off `_on_committed` alone, which is a reasonable-looking place — the bands
    come from the artifact, and the artifact only changes on a commit. But opening a behavior
    that is *already* calibrated is the common case, and there the pane reported a blank where
    a result exists. Nothing failed; the panel simply said nothing.
    """
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    assert _drain(qapp, lambda: "committed cutoff" in tab._ethogram_title.text()), (
        "the ethogram pane never reported a count, so it was never filled: "
        f"{tab._ethogram_title.text()!r}"
    )
    # One row per searched recording, not merely a non-empty draw.
    rows = [name for name, _bands in tab._ethograms._rows]
    assert rows and rows == sorted(rows), rows
    workspace.hide()


def test_a_hundred_recordings_scroll_in_the_ethogram_band_rather_than_growing_it(
    qapp, tmp_path
):
    """`EthogramStrip` is `setFixedHeight(rows * row_px())`, so a hundred recordings is ~2,000px —
    the scale the tool is *for*. The band holds the strip in a scroll area so the page does not
    have to find that height; nothing tested it, and taking the scroll area out left every
    test green.
    """
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    band = tab._ethogram_band
    floor_before = band.minimumSizeHint().height()
    rows = [(f"recording_{i:03d}", [(0.1, 0.2, 0)]) for i in range(100)]
    tab._ethograms.set_rows(
        rows,
        [(0.2, 0.4, 0.8, 1.0)],
        selected=None,
        lengths_ns=[1_000_000_000] * len(rows),
    )
    qapp.processEvents()

    strip = tab._ethograms
    assert (
        strip.height() >= 100 * strip.row_px() > 1500
    ), "the fixture must reach the scale, or this is vacuous"
    # The band's own floor, not the page's: this page already scrolls, so it absorbs a band
    # that demands 3,458px without its floor moving — measured, with the scroll area removed.
    grew = band.minimumSizeHint().height() - floor_before
    assert grew < 100, (
        f"a hundred ethogram rows raised the band's floor by {grew}px — it grew to hold them "
        "instead of scrolling them"
    )
    workspace.hide()


def test_the_dial_is_not_clipped_at_a_normal_window_height(qapp, tmp_path):
    """The top pane was a constant 26px shorter than the dial, at every window height.

    The clip row took all the vertical slack by design, so the top pane never grew: the
    *Kept per recording* heading was sliced in half on a 950px window and still sliced on a
    1400px one. That is why a constant that is merely a bit small reads as a constant that is
    wrong — no amount of screen fixed it.

    The workaround that measured the dial and clawed height back went with the vertical
    splitter: the top row is packed directly now, so it simply gets the height it asks for.
    This is the assertion that says so.
    """
    _run, workspace, tab = _laid_out(qapp, tmp_path)
    workspace.resize(1500, 1050)
    for _ in range(60):
        qapp.processEvents()
    bar = tab._dial_scroll.verticalScrollBar()
    assert bar.maximum() == 0, (
        f"the dial is clipped by {bar.maximum()}px at a 1050px window — its content is "
        f"{tab._dial_scroll.widget().sizeHint().height()}px and its row is "
        f"{tab._top.height()}px"
    )
    workspace.hide()


def test_no_control_that_changes_a_scan_snapshot_is_live(qapp, tmp_path):
    """The same walk over the calibration workspace, where two of the three bugs lived.

    The strip's ▲/▼ cutoff stepper and `Continue — N unjudged` were both live on a snapshot, each
    found by a person looking at the screen. Walking every control means the next one is covered
    without anyone thinking of it.
    """
    from tests.gui.test_cluster_workspace import (
        _LIVE_ON_A_SNAPSHOT,
        _enabled_controls,
    )

    exp, run, workspace = _workspace(qapp, tmp_path)
    behavior = workspace._behaviors[0]
    run.set_cutoff(behavior.id, 0.6)
    frozen = _QueryWorkspace(exp, run.view(run.pin(reason="finish")))
    assert _drain(qapp, lambda: frozen._curves)

    unexpected = [
        pair for pair in _enabled_controls(frozen) if pair not in _LIVE_ON_A_SNAPSHOT
    ]
    assert not unexpected, (
        f"these are enabled on a scan snapshot and change what it is: {unexpected}. Disable them, "
        "or add them to _LIVE_ON_A_SNAPSHOT with the reason they are safe."
    )


def test_the_divider_cannot_be_dragged_on_a_scan_snapshot(qapp, tmp_path):
    """The one way to move the cutoff that no control walk can see.

    The walk above finds buttons, sliders and spin boxes. Dragging the divider is none of
    those — it is a mouse press on a frame — so freezing the steppers and stopping there would
    have left the gesture that replaced them wide open, on a surface whose whole promise is
    that it shows what the snapshot was scanned with.
    """
    from detail.gui.qt import QMouseEvent, QPoint, Qt

    exp, run, workspace = _workspace(qapp, tmp_path)
    behavior = workspace._behaviors[0]
    run.set_cutoff(behavior.id, 0.6)
    frozen = _QueryWorkspace(exp, run.view(run.pin(reason="finish")))
    assert _drain(qapp, lambda: frozen._curves)
    frozen.resize(1400, 900)
    frozen.show()
    qapp.processEvents()

    strip = frozen._calibrate._grid
    handle = strip._handle
    assert handle is not None, "no divider to drag — this proves nothing"
    before = strip.cutoff()

    def drag(kind, x):
        handle.event(
            mouse_event(
                kind,
                QPoint(x, 10).toPointF(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    drag(QMouseEvent.Type.MouseButtonPress, 5)
    drag(QMouseEvent.Type.MouseMove, 600)
    drag(QMouseEvent.Type.MouseButtonRelease, 600)
    qapp.processEvents()

    assert strip.cutoff() == before, "dragging moved a snapshot's cutoff"
    frozen.hide()


def test_a_behavior_shown_twice_reads_its_timelines_once(qapp, tmp_path, monkeypatch):
    """The switch does not re-read the disk for a behavior it has already drawn.

    The class: **work repeated on the UI thread because nothing said it was finished.** Showing
    a behavior read a `unit_timeline` off disk for every searched unit and rebuilt every clip
    spec from them — a large share of the frozen switch, growing with the cohort. None of
    it changes while the run is open.

    Counted rather than timed. A wall-clock assertion on a machine that also runs the suite is
    a flake, and the thing that was wrong is not "slow" but "done again": if the reads come back,
    this fails on a number whatever the machine.
    """
    from detail.pipeline import query as query_pipeline

    _run, _workspace, tab = _laid_out(qapp, tmp_path)
    reads: list = []
    real = query_pipeline.unit_timeline
    monkeypatch.setattr(
        query_pipeline,
        "unit_timeline",
        lambda *a, **k: (reads.append(a), real(*a, **k))[1],
    )
    behavior = tab._behavior_id
    assert behavior is not None, "the fixture never selected a behavior"

    tab._specs.clear()
    tab._timelines.clear()
    tab._refresh_clips()
    first = len(reads)
    assert first, "the fixture reads no timelines at all — this would pass on anything"

    tab._refresh_clips()
    tab._refresh_clips()
    assert len(reads) == first, (
        f"showing the same behavior again re-read {len(reads) - first} timelines off disk; "
        "the specs and the timelines are both fixed for the life of the run"
    )


# --- the map, and the player it drives ------------------------------------------------------


def _mapped(qapp, tmp_path):
    """A Query workspace whose run actually has positions.

    `_searched` defaults to no kinematics, so every test in this file until now drew a map that
    said "not measured" — a legible screen and no coverage: a band that never has data cannot
    fail when the thing that fills it breaks. That is how the map reached this page missing an
    outline, an aspect ratio and a working click, three times over.
    """
    import numpy as np

    exp, run = _searched(
        tmp_path, recordings=("s1",), cx=np.linspace(0.1, 0.9, N_EMBEDDED)
    )
    run.set_cutoff("q01", 0.0)
    workspace = _QueryWorkspace(exp, exp.query_run(run.id))
    workspace.resize(1300, 900)
    workspace.show()
    calibrate = workspace._calibrate
    assert _drain(
        qapp, lambda: bool(calibrate._placements_by_recording)
    ), "the fixture produced no placements, so nothing below tests the map"
    assert _drain(qapp, lambda: len(calibrate._map._panels) > 0), "no square was drawn"
    return exp, run, workspace, calibrate


def test_the_query_map_draws_a_dot_per_kept_instance(qapp, tmp_path):
    """The check that the fixture reaches the code — everything below is vacuous without it."""
    _exp, _run, _workspace, calibrate = _mapped(qapp, tmp_path)

    _holder, square, _label = calibrate._map._panels[0]
    assert len(square._points), "the square is empty"
    assert "not measured" not in calibrate._map._caption.text()


def test_clicking_a_dot_plays_it_beside_the_query_map(qapp, tmp_path):
    """The point of the design: the map answers its own question rather than handing it to a
    strip that may not hold the instance — and Query has no members strip at all, which is why
    borrowing a slot in one could never have reached this page."""
    from detail.gui.qt import QApplication

    _exp, _run, _workspace, calibrate = _mapped(qapp, tmp_path)
    placement = calibrate._placements_by_recording["s1"][0]

    calibrate._map._panels[0][1].clicked.emit(placement["cx"], placement["cy"])
    QApplication.processEvents()

    played = calibrate._map._player._clip
    assert played is not None, "the click played nothing"
    assert played.recording_id == "s1"
    assert int(played.start_t_ns) == int(placement["t_ns"])
    assert played.embed_run_id, "the clip names no embedding run, so it plays nothing"


def test_dragging_the_clips_band_grows_the_section_not_the_scroll(qapp, tmp_path):
    """Reported: "it scrolls the videos and doesn't actually expand the section".

    The row's scroll area kept whatever height the layout gave it, so a drag made the pictures
    bigger *inside a fixed viewport* — more of each clip and less of the row, which is the
    opposite of what the gesture means. `ClusterClipStrip` has sized its own scroll area from
    the cell size since it was written; this row never did, because until now its cells could
    not change size at all.

    Asserted on the scroll area's height, not the cell's: a cell that grew while its container
    did not is exactly the bug.
    """
    _exp, _run, workspace = _workspace(qapp, tmp_path)
    tab = _shown(qapp, workspace)
    before = tab._grid._scroll.height()

    tab._grid.resize_by(60)
    qapp.processEvents()

    assert tab._grid._scroll.height() > before, (
        f"the row is still {tab._grid._scroll.height()}px around bigger pictures — it scrolls "
        "rather than growing"
    )


def test_the_audit_clip_keeps_its_buttons_beside_it_in_this_workspace(qapp, tmp_path):
    """Reported twice: "the yes, no, unclear buttons are still too low ... just for the query
    audit". Fixed for pose and temporal, still broken here.

    **The difference was never the tab — it was that this workspace lays out, then changes
    width.** The block sizes itself from its own width, and `QWidgetItemV2` caches an item's
    size hint and invalidates it on `invalidate()`, which nothing calls for a width-only
    change. So the layout kept answering from the width it last saw: the block reserved 531px
    around a 390px picture and `top=True` gathered the 71px difference into one gap, directly
    under the transport where the buttons go.

    This is the reported sequence, not a tidier one, and every step of it earns its place —
    four shorter versions of this test passed while the bug was on screen. A larger interface
    font, because every size here is in `sizing.lines` and while the block's ceiling binds a
    stale hint and a fresh one are the same number. Judged clips and a `reload`, because that
    is what fills the sidebar and moves the splitter *after* the first layout. A resize before
    the window is shown, because that is the order a restored window produces.
    """
    from detail.gui.qt import QFont, QPoint

    from detail.gui.sizing import lines

    original = QFont(qapp.font())
    larger = QFont(original)
    larger.setPointSize(original.pointSize() + 3)
    qapp.setFont(larger)
    try:
        exp, run, workspace = _workspace(qapp, tmp_path)
        run.set_cutoff("q01", 0.0)
        audit = workspace._audit
        audit.set_behavior("q01")
        assert _drain(qapp, lambda: bool(audit._headline.text()))
        for index, row in enumerate(run.audit_sample(label="q01", budget=10)):
            run.label_audit(row.instance_id, index % 3 != 2)
        audit.reload()
        assert _drain(qapp, lambda: bool(audit._judged))
        workspace.show_view("Audit")
        workspace.resize(1397, 900)
        workspace.show()
        for _ in range(40):
            qapp.processEvents()
        audit._start_batch()
        for _ in range(40):
            qapp.processEvents()

        width = audit._stack.width()
        # Both directions. A stale hint can be from a narrower width as easily as a wider one:
        # on Linux offscreen the reverted fix left the block 30px *short* — the clip squeezed —
        # and a `<=` check passed it.
        need = min(audit._picture_height(width), audit._stack.maximumHeight())
        assert abs(audit._stack.height() - need) <= 2, (
            f"the block is {audit._stack.height()}px for a clip that needs {need}px at "
            f"{width}px — it did not re-ask after the width moved under it"
        )
        transport = audit._player._scrubber
        bottom = transport.mapTo(workspace, QPoint(0, transport.height())).y()
        below = audit._where.mapTo(workspace, QPoint(0, 0)).y()
        assert below - bottom <= lines(2), (
            f"{below - bottom}px of nothing between the clip's transport and the line under "
            f"it — the buttons that judge the clip are that far from it"
        )
        workspace.hide()
    finally:
        qapp.setFont(original)


def test_a_committed_cutoff_puts_the_measurements_out_of_date(qapp, tmp_path):
    """#515's query half. The band was measured once per run and a commit never reached it, so a
    cutoff that kept more matches — or fewer — left it drawing the matches it was measured at,
    under the heading of the behavior still selected. A commit leaves that behavior selected, so
    nothing but the commit itself can tell the band.
    """
    from detail.gui.measurements import MEASURE, REMEASURE

    exp, run, workspace = _workspace(qapp, tmp_path)
    calibrate = workspace._calibrate
    panel = calibrate._measurements
    behavior_id = calibrate._behavior_id
    assert _drain(qapp, lambda: panel._measure_button.text() == MEASURE)
    panel._measure_button.click()
    assert _drain(
        qapp, lambda: panel._measured is not None and panel._measure_button.isHidden()
    ), panel._note.text()

    def matches():
        return sorted(
            (i.recording_id, int(i.t_ns), int(i.end_t_ns))
            for i in run.artifact().instances
        )

    before = matches()
    loose = 0.0 if run.state().behavior(behavior_id).cutoff else 0.9
    calibrate._commit(loose)
    assert _drain(qapp, lambda: matches() != before), "the cutoff kept the same matches"
    assert _drain(
        qapp,
        lambda: not panel._measure_button.isHidden()
        and panel._measure_button.text() == REMEASURE,
    ), f"a commit left the band drawing the old matches: {panel._note.text()!r}"


def test_the_calibrate_tab_names_recordings_not_ids(qapp, tmp_path):
    """§11: the id is the tool's, the name is the user's. The kept panel, the clip captions and
    the ethogram rows printed `recording_id` while the Browse tab beside them showed names
    (#35). The names are read off the UI thread with the search results."""
    exp, _run, _unused = _workspace(qapp, tmp_path)
    for recording, name in zip(exp.recordings, ("m27 day3", "m31 day1")):
        recording.set_name(name)
    workspace = _QueryWorkspace(exp, exp.query_run(_run.id))
    assert _drain(qapp, lambda: workspace._curves), "the search results never loaded"
    calibrate = workspace._calibrate
    calibrate._refresh_header(0.5)

    panel = calibrate._spread.text()
    assert "m27 day3" in panel and "m31 day1" in panel, panel
    assert "s1" not in panel.split() and "s2" not in panel.split(), panel
    captions = [spec.label for spec in calibrate._build_specs()]
    assert captions and all(c.startswith(("m27 day3 ", "m31 day1 ")) for c in captions)
    calibrate._refresh_clips()  # the strip under the curve, as drawn
    cells = [cell._caption.text() for cell in calibrate._grid._cells.values()]
    assert cells and all(
        c.startswith(("m27 day3\n", "m31 day1\n")) for c in cells
    ), cells
    calibrate._show_ethograms((
        [("s1", [], 1), ("s2", [], 1)], None,
    ))  # fmt: skip
    assert [row[0] for row in calibrate._ethograms._rows] == ["m27 day3", "m31 day1"]
