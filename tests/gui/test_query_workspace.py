"""The §20 calibration workspace — the surface where a cutoff is actually chosen.

What is checked here is what a construction smoke test would pass straight through: that
moving the cutoff **rewrites nothing** and only **Keep** saves it, that the page says whether
the cutoff is set, that the grid splits every clip at the cutoff in words as well as fades, and
that the counts on screen are the ones a save writes.

The old CALIBRATE — a one-row strip of still frames beside a player, saved by letting go of a
drag — had its own guards here (decode, recentring, the divider, *Cut here*, the ethogram band).
They went with it, and so did the strip's widget (`clip_grid.ConfidenceStrip`) and its own
guards; what it shared with the cluster strip is still guarded in `test_decode_priority.py`.

The run is real — the fixture runs the actual search over synthetic embeddings with a planted
motif, reusing `tests/api/test_query.py`'s helper. Frame decode is not: this fixture has no
video, which is what makes the decode-failure guard natural rather than contrived.
"""

import math
import re

import numpy as np
import pytest

from detail.gui import frame_source
from detail.gui.clip_grid import score_digits
from detail.gui.modules import analysis_query
from detail.gui.modules.analysis_query import (
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


def test_moving_the_cutoff_rewrites_nothing(qapp, tmp_path):
    """§20's own guard, for every way the cutoff moves: the line, the number and a clicked clip.

    Reprojection rewrites a frame-view zarr per searched unit, so doing it per pixel of a drag
    freezes the GUI — and now a move is a *preview*: Keep is the only write. The state is checked
    as well as the file: `set_cutoff` writes `query.yaml` before it reprojects, so a move that
    saved would show there even while its reprojection was still running.
    """
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.5)
    calibrate = _tiled(qapp, workspace)
    before = (run._run_dir() / "instances.csv").read_text(encoding="utf-8-sig")

    for value in (0.6, 0.7, 0.8, 0.9):
        calibrate._line.setValue(value)  # fires sigPositionChanged, as a drag does
        qapp.processEvents()
    calibrate._on_confidence_typed(0.85)
    calibrate._montage.clip_clicked.emit(2, False)
    for _ in range(50):  # give an errant write every chance to land
        qapp.processEvents()

    assert run.state().behavior("q01").cutoff == 0.5
    assert (run._run_dir() / "instances.csv").read_text(encoding="utf-8-sig") == before
    # ...and the view did move, so this is not passing because nothing happened at all
    shown = calibrate.cutoff()
    assert shown != 0.5
    assert (
        f"You're looking at {analysis_query._cutoff_text(shown)}"
        in calibrate._header.text()
    )


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
    without the guard one keystroke refreshes twice, and each refresh re-sections the grid and
    recounts every recording.
    """
    exp, run, workspace = _workspace(qapp, tmp_path)
    calibrate = workspace._calibrate
    calls = []
    original = calibrate._preview
    calibrate._preview = lambda value, **k: (calls.append(value), original(value, **k))[
        1
    ]

    calibrate._on_confidence_typed(0.66)
    assert len(calls) == 1, f"one keystroke refreshed {len(calls)} times"


# --- CALIBRATE: every clip, split at the cutoff, and one button that saves it ----------------


def _tiled(qapp, workspace):
    """Calibrate with its grid filled: the preview-cache read is off the UI thread."""
    calibrate = workspace._calibrate
    assert _drain(
        qapp, lambda: calibrate._tiles_read and calibrate._montage.clips()
    ), "the grid never filled"
    return calibrate


def test_an_unset_cutoff_says_so_and_offers_keep(qapp, tmp_path):
    """The reported dead end: AUDIT asked for a cutoff, CALIBRATE showed a slider at 0.5 and a
    line reading "not calibrated", and the one button that looked like setting it was grey.

    Now the page says the cutoff is not set and offers the save by name; after it, the page
    says it is saved and the button has nothing left to do. The negative half: a saved cutoff
    must not read as unset.
    """
    exp, run, workspace = _workspace(qapp, tmp_path)
    calibrate = _tiled(qapp, workspace)
    assert run.state().behavior("q01").cutoff is None
    assert calibrate._header.text().startswith("⚠ ")
    assert "Not saved" in calibrate._header.text()
    assert "has no cutoff yet" in calibrate._header.text()
    assert calibrate._keep.isEnabled()
    assert calibrate._keep.text() == "Keep clips from 0.5 up"

    calibrate._keep.click()
    assert _drain(qapp, lambda: run.state().behavior("q01").cutoff == 0.5)
    assert _drain(qapp, lambda: calibrate._header.text().startswith("✓ "))
    assert "saved" in calibrate._header.text()
    assert not calibrate._keep.isEnabled()
    assert calibrate._keep.toolTip() == "This is the saved cutoff."


def test_keep_saves_the_shown_cutoff_once(qapp, tmp_path):
    """A preview, then Keep: one write, of the cutoff on screen, and every other tab told."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.5)
    calibrate = _tiled(qapp, workspace)
    heard = []
    calibrate.committed.connect(lambda: heard.append(True))
    writes = []
    real = workspace._run.set_cutoff
    workspace._run.set_cutoff = lambda *a, **k: (writes.append(a), real(*a, **k))[1]

    calibrate._on_confidence_typed(0.72)
    qapp.processEvents()
    assert writes == [], "a typed cutoff saved itself"
    assert "still use 0.5" in calibrate._header.text()
    assert calibrate._keep.text() == "Keep clips from 0.72 up"

    calibrate._keep.click()
    assert _drain(qapp, lambda: heard), "the other tabs were never told"
    assert [a[1] for a in writes] == [pytest.approx(0.72)]
    assert run.state().behavior("q01").cutoff == pytest.approx(0.72)


def test_a_labelled_rule_sits_at_the_cutoff_and_nothing_below_it_fades(qapp, tmp_path):
    """Every candidate, most confident first; a dashed rule after the last clip kept says, in
    words, how many are kept and dropped and at what cutoff. Nothing below it is faded: those
    clips are still to be watched (they were faded and tagged *dropped*, and could not be).
    """
    exp, run, workspace = _workspace(qapp, tmp_path)
    calibrate = _tiled(qapp, workspace)
    calibrate._on_confidence_typed(0.4)
    specs = calibrate._behavior_specs()
    scores = [float(s.score) for s in specs]
    assert scores == sorted(scores, reverse=True), "the grid is not the ranking"
    kept = sum(1 for v in scores if v >= 0.4)
    assert 0 < kept < len(specs), "the fixture must put clips on both sides of 0.4"

    montage = calibrate._montage
    assert montage._rule_headings, "the cutoff is not drawn as a rule"
    (untitled, first), (rule, rest) = montage.sections()
    assert untitled == "" and (first, rest) == (kept, len(specs) - kept)
    assert f"{kept:,} kept" in rule and "cutoff 0.4" in rule
    assert f"{len(specs) - kept:,} dropped" in rule
    tiles = montage.clips()
    assert len(tiles) == len(specs)
    assert not any(t.faded for t in tiles)
    assert {t.tag for t in tiles} <= {None, "your example", "your non-example"}


def test_a_clicked_clip_stays_where_it_was_clicked(qapp, tmp_path):
    """The rows re-flow when the rule moves, so a click scrolls the grid to keep the clip it
    chose at the height it was clicked at — what is under the pointer is still that clip
    (reported as clunky: the click re-flowed the rows and another clip took its place).
    """
    exp, run, workspace = _workspace(qapp, tmp_path)
    workspace.resize(1400, 700)
    workspace.show()
    calibrate = _tiled(qapp, workspace)
    montage = calibrate._montage
    specs = calibrate._behavior_specs()
    calibrate._on_confidence_typed(float(specs[2].score))
    qapp.processEvents()
    # A dropped clip a few rows past the rule, so the grid has room above it to hold it still.
    index = montage.sections()[0][1] + 3 * montage._columns() + 2
    assert index < len(specs), "the fixture has too few clips past the rule"
    scroll = montage.grid.verticalScrollBar()
    scroll.setValue(max(montage.tile_rect(index).top() - 100, 0))
    qapp.processEvents()
    on_screen = montage.tile_rect(index).top() - scroll.value()

    calibrate._on_tile_clicked(index, False)
    qapp.processEvents()
    assert (
        montage.sections()[0][1] == index + 1
    ), "the clicked clip is not the last kept"
    assert (
        montage.tile_rect(index).top() - scroll.value() == on_screen
    ), "the clicked clip moved on screen"
    workspace.hide()


def test_your_own_example_is_tagged_where_it_is_found(qapp, tmp_path):
    """The search finds the clips it was given, at the top of the ranking. Unmarked, they read as
    its most confident finds; they are tagged *your example* (or *your non-example*), by the one
    rule AUDIT uses to set them aside (`query_runs.told_kind`)."""
    from detail.storage import query_runs

    exp, run, workspace = _workspace(qapp, tmp_path)
    calibrate = _tiled(qapp, workspace)
    behavior = run.state().behavior("q01")
    told = query_runs.told_kinds([behavior])
    expected = [
        query_runs.told_kind(
            told, "q01", s.recording_id, s.track_id, s.start_t_ns, s.end_t_ns
        )
        for s in calibrate._behavior_specs()
    ]
    assert (
        query_runs.EXAMPLE in expected
    ), "no candidate sits on an example — proves nothing"
    tags = [t.tag for t in calibrate._montage.clips()]
    assert tags == [None if k is None else f"your {k}" for k in expected]


def test_scrolled_below_the_line_a_bar_says_so_and_goes_back(qapp, tmp_path):
    """Below the line the clips look like any other, so a bar pinned over the grid says they are
    dropped, and its button scrolls back to the line."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    workspace.resize(1400, 700)
    workspace.show()
    calibrate = _tiled(qapp, workspace)
    montage = calibrate._montage
    specs = calibrate._behavior_specs()
    # A cutoff near the top of the ranking, so most of the grid lies below the line.
    calibrate._on_confidence_typed(float(specs[min(3, len(specs) - 2)].score))
    qapp.processEvents()
    bar = calibrate._below_bar
    scroll = montage.grid.verticalScrollBar()
    scroll.setValue(0)
    qapp.processEvents()
    assert not bar.isVisible(), "the bar shows while the line is in view"

    scroll.setValue(scroll.maximum())
    qapp.processEvents()
    assert scroll.maximum() > 0, "the grid does not scroll, so this proves nothing"
    assert bar.isVisible(), "scrolled past the line and nothing says so"
    assert "Below the cutoff" in bar.text() and "dropped" in bar.text()
    assert bar.y() == 0 and bar.parent() is montage.grid.viewport()

    bar.back.click()
    qapp.processEvents()
    lowest = montage.heading_rects()[-1][1].bottom()
    top = scroll.value()
    assert (
        top <= lowest <= top + montage.grid.viewport().height()
    ), "the line is not in view"
    assert not bar.isVisible()
    workspace.hide()


def test_a_clicked_clip_becomes_the_last_one_kept(qapp, tmp_path):
    """Clicking a clip shows the cutoff at it — previewed, not saved."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    calibrate = _tiled(qapp, workspace)
    specs = calibrate._behavior_specs()
    index = min(6, len(specs) - 1)

    calibrate._montage.clip_clicked.emit(index, False)
    qapp.processEvents()
    assert calibrate.cutoff() == pytest.approx(float(specs[index].score), abs=1e-6)
    kept = calibrate._montage.sections()[0][1]
    assert kept >= index + 1, "the clicked clip is not kept"
    assert run.state().behavior("q01").cutoff is None, "a click saved the cutoff"


def test_a_clip_removed_in_audit_is_tagged_on_calibrate(qapp, tmp_path):
    """The review's removals show where the cutoff is chosen: the clip stays in the ranking
    (it is the method's), tagged *removed* and crossed, and the header counts it."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.0)
    clip = run.review_clips("q01", "s1")[0]
    run.take_out_clips([clip], from_label="q01")
    workspace._calibrate.forget_clips()
    calibrate = _tiled(qapp, workspace)
    calibrate._refresh_header(0.0)
    calibrate._show_tiles()

    marked = [t for t in calibrate._montage.clips() if t.tag == "removed"]
    assert len(marked) == 1 and marked[0].crossed and marked[0].faded
    assert "1 of them removed in Audit" in calibrate._header.text()


def test_an_unsaved_cutoff_is_loud_on_calibrate_and_quiet_once_kept(qapp, tmp_path):
    """Reported: a previewed cutoff read as applied, and AUDIT then reviewed the saved one. While
    the cutoff on screen is not the saved one the bar is banded, bold and amber with *Not saved*
    in words, the line on the curve is dashed and the rule says *(not saved)*; once kept, all
    three go quiet. The negative half is the saved state, so the loud one is not just the look.
    """
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.5)
    workspace._calibrate.set_behavior("q01", workspace._curves["q01"])
    calibrate = _tiled(qapp, workspace)
    bar = calibrate._save_bar

    def loud() -> bool:
        return (
            bar.autoFillBackground() and bar._row is not None and bar.text.font().bold()
        )

    def dashed() -> bool:
        from detail.gui.qt import Qt

        return calibrate._line.pen.style() == Qt.PenStyle.DashLine

    def rule() -> str:
        headings = calibrate._montage.heading_rects()
        return headings[-1][0] if headings else ""

    assert (
        not loud() and not dashed() and "not saved" not in rule()
    ), "the saved state is loud"
    assert calibrate._header.text().startswith("✓ ")
    assert not calibrate._back.isVisible() and not calibrate._back.isEnabled()

    calibrate._on_confidence_typed(0.8)
    qapp.processEvents()
    assert loud() and dashed() and "(not saved)" in rule()
    assert calibrate._header.text().startswith("⚠ Not saved")
    assert calibrate._back.isEnabled() and calibrate._back.text() == "Back to 0.5"

    calibrate._keep.click()
    assert _drain(qapp, lambda: run.state().behavior("q01").cutoff == 0.8)
    assert _drain(qapp, lambda: not loud())
    assert not dashed() and "not saved" not in rule()


def test_back_to_the_saved_cutoff_writes_nothing(qapp, tmp_path):
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.5)
    workspace._calibrate.set_behavior("q01", workspace._curves["q01"])
    calibrate = _tiled(qapp, workspace)
    calibrate._on_confidence_typed(0.8)
    qapp.processEvents()
    before = (run._run_dir() / "query.yaml").read_text(encoding="utf-8-sig")

    calibrate._back.click()
    qapp.processEvents()
    assert calibrate.cutoff() == pytest.approx(0.5)
    assert calibrate.unsaved() is None
    assert (run._run_dir() / "query.yaml").read_text(encoding="utf-8-sig") == before


def test_the_other_views_say_calibrate_holds_an_unsaved_cutoff(qapp, tmp_path):
    """Where the mistake was found: on AUDIT, reviewing the saved cutoff while CALIBRATE showed
    another. The notice names both and offers Keep; Keep from it saves and the notice goes. Not on
    CALIBRATE itself, which has its own bar, and not while nothing is unsaved."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.5)
    workspace._calibrate.set_behavior("q01", workspace._curves["q01"])
    calibrate = _tiled(qapp, workspace)
    notice = workspace._unsaved_notice

    for view in ("Audit", "Browse"):
        workspace.show_view(view)
        qapp.processEvents()
        assert notice.isHidden(), f"{view} warns with nothing unsaved"

    workspace.show_view("Calibrate")
    calibrate._on_confidence_typed(0.8)
    qapp.processEvents()
    assert (
        notice.isHidden()
    ), "CALIBRATE has its own bar; the notice is for the other views"
    for view in ("Audit", "Browse"):
        workspace.show_view(view)
        qapp.processEvents()
        assert not notice.isHidden(), f"{view} says nothing about the unsaved cutoff"
        assert "not saved" in notice.text.text() and "0.8" in notice.text.text()
        assert "0.5" in notice.text.text()

    workspace._notice_keep.click()
    assert _drain(qapp, lambda: run.state().behavior("q01").cutoff == 0.8)
    assert _drain(qapp, lambda: notice.isHidden())

    calibrate._on_confidence_typed(0.6)
    qapp.processEvents()
    workspace.show_view("Audit")
    qapp.processEvents()
    workspace._notice_to_calibrate.click()
    qapp.processEvents()
    assert workspace.current_view() == "Calibrate"


def test_the_below_cutoff_bar_is_centred_and_stands_out(qapp, tmp_path):
    """Reported: the button sat at the far right and the bar was the page's own colour, lost
    over the tiles in a dark theme. Its words and button are centred together, and it is a
    framed band in a shade the page is not, its words in an amber that reads on that band.
    """
    from detail.gui.qt import QPalette
    from detail.gui.qt import QFrame

    from detail.gui.status_color import warn_ink

    exp, run, workspace = _workspace(qapp, tmp_path)
    workspace.resize(1400, 700)
    workspace.show()
    calibrate = _tiled(qapp, workspace)
    specs = calibrate._behavior_specs()
    calibrate._on_confidence_typed(float(specs[min(3, len(specs) - 2)].score))
    scroll = calibrate._montage.grid.verticalScrollBar()
    scroll.setValue(scroll.maximum())
    qapp.processEvents()
    bar = calibrate._below_bar
    assert bar.isVisible()

    left = bar._text.geometry().left()
    right = bar.width() - bar.back.geometry().right()
    assert abs(left - right) <= 4, f"not centred: {left}px left, {right}px right"
    assert bar.frameShape() == QFrame.Shape.Box and bar.autoFillBackground()
    page = bar.parentWidget().palette().color(QPalette.ColorRole.Window)
    assert (
        bar.palette().color(QPalette.ColorRole.Window) != page
    ), "the page's own colour"
    assert bar._text.palette().color(QPalette.ColorRole.WindowText) == warn_ink(bar)
    workspace.hide()


def test_the_warning_amber_reads_on_a_dark_band(qapp):
    """``WARN`` is a dark amber: on a light band it reads, on a dark one it does not, so a band
    lightens it there."""
    from detail.gui.qt import QColor, QPalette
    from detail.gui.qt import QWidget

    from detail.gui.status_color import WARN, warn_ink

    widget = QWidget()
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#eeeeee"))
    widget.setPalette(palette)
    assert warn_ink(widget) == WARN
    palette.setColor(QPalette.ColorRole.Window, QColor("#3a3a3a"))
    widget.setPalette(palette)
    dark = warn_ink(widget)
    assert dark.lightness() > WARN.lightness() + 30, "no lighter on a dark band"


# --- what the view says -------------------------------------------------------------------


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

    calibrate._refresh_header(0.5)
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
    """The count the header states, e.g. `⚠ rearing: cutoff not set — 0.5 would keep 228 of …`."""
    text = calibrate._header.text()
    return int(re.search(r"keeps? ([\d,]+) of", text).group(1).replace(",", ""))


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
            int(line.split(" ←")[0].split()[-1].replace(",", ""))
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


# --- browse ---------------------------------------------------------------------------------


def _browsed(qapp, workspace):
    """The workspace's BROWSE once its off-thread read has landed with rows in it."""
    browse = workspace._browse
    browse.reload()
    assert _drain(
        qapp, lambda: browse._loaded and browse._model.rowCount() > 0
    ), "BROWSE listed nothing"
    return browse


def test_browse_is_clustering_browse(qapp, tmp_path):
    """One BROWSE for both run kinds (§20): the query page's own table went, and with it a
    second implementation of filters, ethogram, list and player."""
    from detail.gui.cluster_workspace import QUERY_TERMS
    from detail.gui.label_browse import LabelBrowseTab

    exp, run, workspace = _workspace(qapp, tmp_path)
    assert type(workspace._browse) is LabelBrowseTab
    assert workspace._browse._terms is QUERY_TERMS


def test_browse_lists_the_accepted_instances(qapp, tmp_path):
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.9)
    browse = _browsed(qapp, workspace)
    assert browse._model.rowCount() == len(browse._rows)


def test_browse_lists_recordings_by_name(qapp, tmp_path):
    """§11: the id is the tool's, the name is the user's — in the list, not only in pickers."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.0)
    for recording, name in zip(exp.recordings, ("Fish tank A", "Fish tank B")):
        recording.set_name(name)
    browse = _browsed(qapp, workspace)
    shown = {browse._model.item(r, 1).text() for r in range(browse._model.rowCount())}
    assert shown <= {"Fish tank A", "Fish tank B"} and shown, shown


def test_browse_survives_finish(qapp, tmp_path):
    """§20 — finishing sets the record status, it does not close the workspace.
    Reopening is only needed to *change* something."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.9)
    workspace._sync_status()
    browse = _browsed(qapp, workspace)
    listed = browse._model.rowCount()

    workspace._on_finish()
    # Off-thread: finishing pins, and a pin first rebuilds the frame views a cutoff deferred.
    assert _drain(qapp, lambda: run.status == "complete")

    # No re-populate: the rows have to *survive* finishing, not be rebuilt by the assertion.
    assert browse._model.rowCount() == listed
    assert browse.isEnabled()
    assert browse.grab() is not None  # and it still paints


def test_finish_waits_for_every_cutoff_and_says_so(qapp, tmp_path):
    """Finish is greyed, with the reason in words beside it, until every behavior has a saved
    cutoff — the API's own refusal (`QueryRun.finish_blocker`), read before the press. A run
    finished uncut was the dead end AUDIT then described twice at once. Saving through Keep is
    what lifts it, with no other refresh."""
    exp, run, workspace = _workspace(qapp, tmp_path)
    calibrate = _tiled(qapp, workspace)
    assert not workspace._finish.isEnabled()
    assert "no cutoff yet" in workspace._finish.toolTip()
    assert workspace._behaviors[0].name in workspace._finish_note.text()

    calibrate._keep.click()
    assert _drain(
        qapp, lambda: workspace._finish.isEnabled()
    ), "Keep left Finish greyed"
    assert workspace._finish_note.text() == ""


def test_finish_and_reopen_swap_which_button_applies(qapp, tmp_path):
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff(workspace._behaviors[0].id, 0.9)  # finishing needs every cutoff
    workspace._sync_status()  # saved behind the page's back, so it is told
    assert workspace._finish.isEnabled() and not workspace._reopen.isEnabled()
    workspace._on_finish()
    assert _drain(
        qapp, lambda: workspace._reopen.isEnabled()
    )  # finishing is off-thread
    assert not workspace._finish.isEnabled() and workspace._reopen.isEnabled()
    workspace._on_reopen()
    assert workspace._finish.isEnabled() and not workspace._reopen.isEnabled()


def test_a_finished_run_freezes_the_cutoff_until_reopened(qapp, tmp_path):
    """`set_cutoff` and the review both refuse a finished run, and the controls were frozen for
    a snapshot only — so on a finished run the line dragged, the box took a number and a removal
    was written, each into an error. Frozen with the reason that names Reopen, and back after it.
    """
    from detail.gui.query_review import FINISHED

    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff(workspace._behaviors[0].id, 0.0)
    workspace._audit.reload()
    calibrate, audit = workspace._calibrate, workspace._audit
    assert _drain(qapp, lambda: audit._data.get("cutoff") is not None), "never loaded"
    assert audit._reason() == ""
    assert calibrate._confidence.isEnabled() and calibrate._line.movable

    workspace._on_finish()
    assert _drain(qapp, lambda: workspace._reopen.isEnabled())
    assert not calibrate._confidence.isEnabled()
    assert not calibrate._line.movable
    assert not calibrate._montage.is_clickable(), "a clicked clip would still move it"
    assert not calibrate._keep.isEnabled()
    assert "Reopen" in calibrate._confidence.toolTip()
    assert "Reopen" in calibrate._keep.toolTip()
    # The review's own post-finish read sets its reason; Finish then says it.
    assert _drain(qapp, lambda: audit._reason() == FINISHED)
    assert not audit.pane.finish.isEnabled()
    assert "Reopen" in audit.pane.finish.toolTip()

    workspace._on_reopen()
    assert calibrate._confidence.isEnabled() and calibrate._line.movable
    assert calibrate._montage.is_clickable()
    assert _drain(qapp, lambda: audit._reason() == "")


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
        tab._preview(cutoff, scroll=True)
        qapp.processEvents()
        heights.add(tab._plot.height())
        ranges.add(tuple(tuple(axis) for axis in tab._plot.viewRange()))
    assert len(heights) == 1, f"the plot changed height as the cutoff moved: {heights}"
    # The other way a graph "resizes": the axes rescale under a curve that never changed.
    assert (
        len(ranges) == 1
    ), f"the plot's view range moved as the cutoff moved: {ranges}"


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

    The view reorders the model, so a row's identity cannot be its position — a `_show_row` that
    read `self._rows[current.row()]` plays, after any sort, a different instance than the one
    clicked. Silent, and wrong in a way that looks like the player is broken.
    """
    from detail.gui.qt import Qt as _Qt

    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.0)
    browse = _browsed(qapp, workspace)
    assert browse._model.rowCount() > 1

    headers = [
        browse._model.horizontalHeaderItem(c).text()
        for c in range(browse._model.columnCount())
    ]
    browse._table.sortByColumn(
        headers.index("confidence"), _Qt.SortOrder.AscendingOrder
    )
    qapp.processEvents()

    for view_row in range(min(browse._model.rowCount(), 3)):
        expected = browse._row_at(view_row)
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
    """Same rule as every score caption: a fixed 2dp makes every row in a tight band read alike.

    Pinned against `score_digits`, which is the contract, rather than against unconditional
    distinctness, which is not: it caps at 5 decimals so a caption stays a caption.
    """
    exp, run, workspace = _workspace(qapp, tmp_path)
    run.set_cutoff("q01", 0.0)
    browse = _browsed(qapp, workspace)
    headers = [
        browse._model.horizontalHeaderItem(c).text()
        for c in range(browse._model.columnCount())
    ]
    column = headers.index("confidence")
    shown = [
        browse._model.item(r, column).text() for r in range(browse._model.rowCount())
    ]
    values = [r.score for r in browse._rows]
    at_cap = {"—" if v is None else f"{v:.5f}" for v in values}
    assert len(set(shown)) == len(at_cap), f"confidences collide: {sorted(shown)}"
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
    # The other two ways, after the curve's line and the number box: a clicked clip and Keep.
    # Freezing two of the ways once left a third live — the write was refused, but the screen
    # showed a cutoff belonging to neither the snapshot nor the run.
    calibrate = _tiled(qapp, frozen)
    assert not calibrate._montage.is_clickable()
    assert not calibrate._keep.isEnabled() and "snapshot" in calibrate._keep.toolTip()
    before = calibrate.cutoff()
    calibrate._on_tile_clicked(0, False)
    assert calibrate.cutoff() == before, "a clicked clip moved a snapshot's cutoff"
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


def test_a_behavior_shown_twice_reads_its_timelines_once(qapp, tmp_path, monkeypatch):
    """The switch does not re-read the disk for a behavior it has already drawn.

    The class: **work repeated on the UI thread because nothing said it was finished.** Showing
    a behavior read a `unit_timeline` off disk for every searched unit and rebuilt every clip
    spec from them — a large share of the frozen switch, growing with the cohort. None of it
    changes while the run is open, and a cutoff move only re-sections the same clips.

    Counted rather than timed: if the reads come back, this fails on a number whatever the
    machine.
    """
    from detail.pipeline import query as query_pipeline

    _exp, _run, workspace = _workspace(qapp, tmp_path)
    tab = _tiled(qapp, workspace)
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
    tab._show_tiles(new_source=True)
    first = len(reads)
    assert first, "the fixture reads no timelines at all — this would pass on anything"

    tab._preview(0.3, scroll=True)
    tab.set_behavior(behavior, tab._curve)
    tab._show_tiles()
    assert len(reads) == first, (
        f"showing the same behavior again re-read {len(reads) - first} timelines off disk; "
        "the specs and the timelines are both fixed for the life of the run"
    )


# --- the map, and the player it drives ------------------------------------------------------


def _mapped(qapp, tmp_path):
    """A Query workspace whose run actually has positions, opened on MEASURES.

    `_searched` defaults to no kinematics, so a map test on it draws a map that says "not
    measured" — a legible screen and no coverage: a band that never has data cannot fail when
    the thing that fills it breaks. That is how the map reached this page missing an outline, an
    aspect ratio and a working click, three times over.
    """
    import numpy as np

    exp, run = _searched(
        tmp_path, recordings=("s1",), cx=np.linspace(0.1, 0.9, N_EMBEDDED)
    )
    run.set_cutoff("q01", 0.0)
    workspace = _QueryWorkspace(exp, exp.query_run(run.id))
    workspace.resize(1300, 900)
    workspace.show()
    measures = workspace._measures
    workspace.show_view("Measures")
    assert _drain(
        qapp, lambda: bool(measures._placements_by_recording)
    ), "the fixture produced no placements, so nothing below tests the map"
    assert _drain(qapp, lambda: len(measures._map._panels) > 0), "no square was drawn"
    return exp, run, workspace, measures


def test_the_query_map_draws_a_dot_per_kept_instance(qapp, tmp_path):
    """The check that the fixture reaches the code — everything below is vacuous without it."""
    _exp, _run, _workspace, measures = _mapped(qapp, tmp_path)

    _holder, square, _label = measures._map._panels[0]
    assert len(square._points), "the square is empty"
    assert "not measured" not in measures._map._caption.text()


def test_clicking_a_dot_plays_it_beside_the_query_map(qapp, tmp_path):
    """The point of the design: the map answers its own question rather than handing it to a
    strip that may not hold the instance — and Query has no members strip at all, which is why
    borrowing a slot in one could never have reached this page."""
    from detail.gui.qt import QApplication

    _exp, _run, _workspace, measures = _mapped(qapp, tmp_path)
    placement = measures._placements_by_recording["s1"][0]

    measures._map._panels[0][1].clicked.emit(placement["cx"], placement["cy"])
    QApplication.processEvents()

    played = measures._map._player._clip
    assert played is not None, "the click played nothing"
    assert played.recording_id == "s1"
    assert int(played.start_t_ns) == int(placement["t_ns"])
    assert played.embed_run_id, "the clip names no embedding run, so it plays nothing"


def test_a_committed_cutoff_puts_the_measurements_out_of_date(qapp, tmp_path):
    """#515's query half. The band was measured once per run and a commit never reached it, so a
    cutoff that kept more matches — or fewer — left it drawing the matches it was measured at,
    under the heading of the behavior still selected. A commit leaves that behavior selected, so
    nothing but the commit itself can tell the band.
    """
    from detail.gui.measurements import MEASURE, REMEASURE

    exp, run, workspace = _workspace(qapp, tmp_path)
    calibrate = workspace._calibrate
    panel = workspace._measures._measurements
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
    """§11: the id is the tool's, the name is the user's. The kept panel and the clip captions
    printed `recording_id` while the Browse tab beside them showed names (#35). The names are
    read off the UI thread with the search results."""
    exp, _run, _unused = _workspace(qapp, tmp_path)
    for recording, name in zip(exp.recordings, ("m27 day3", "m31 day1")):
        recording.set_name(name)
    workspace = _QueryWorkspace(exp, exp.query_run(_run.id))
    assert _drain(qapp, lambda: workspace._curves), "the search results never loaded"
    calibrate = _tiled(qapp, workspace)
    calibrate._refresh_header(0.5)

    panel = calibrate._spread.text()
    assert "m27 day3" in panel and "m31 day1" in panel, panel
    assert "s1" not in panel.split() and "s2" not in panel.split(), panel
    captions = [clip.caption for clip in calibrate._montage.clips()]
    assert captions and all(
        c.endswith((" · m27 day3", " · m31 day1")) for c in captions
    ), captions
