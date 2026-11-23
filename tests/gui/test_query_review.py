"""AUDIT on a query run: every clip a behavior's cutoff keeps, one recording at a time (§20).

Built through the workspace's own tab on a real searched run. The fixture has no previews, so
``Pictures`` stands in for a built preview cache wherever a review is finished.
"""

import dataclasses

import pytest

from detail.gui.qt import QAbstractButton, QLabel
from detail.gui.query_review import FROZEN, QueryReviewTab
from detail.storage import analysis_runs as analysis_store
from tests.api.test_query import _searched
from tests.gui.conftest import Pictures, drain_busy_calls
from tests.gui.test_query_workspace import _drain


def _cutoff_keeping(run, k: int) -> float:
    pool = run._audit_pool("q01")
    examples = run._audit_examples(pool)
    scores = sorted(
        (
            r.score
            for r in pool
            if r.recording_id == "s1" and r.instance_id not in examples
        ),
        reverse=True,
    )
    return scores[k - 1]


@pytest.fixture
def searched(tmp_path):
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", _cutoff_keeping(run, 6))
    return exp, run


def _tab(qapp, exp, run) -> QueryReviewTab:
    tab = QueryReviewTab(exp, run)
    tab.resize(1200, 760)
    tab.show()
    tab.set_behavior("q01")
    assert _drain(qapp, lambda: tab._list.topLevelItemCount() == 2), "never loaded"
    return tab


def _open(qapp, tab, rid: str = "s1") -> None:
    tab.open_recording(rid)
    assert _drain(qapp, lambda: tab._open is not None and tab._open["rid"] == rid)
    montage = tab.pane.montage
    montage.set_source(Pictures(), keep_pictures=True)
    bar = montage.grid.verticalScrollBar()

    def at_end() -> bool:
        bar.setValue(bar.maximum())
        return tab.pane.reached_end() and montage.unpictured() == (0, 0)

    assert _drain(qapp, at_end)


def _pane_text(tab) -> str:
    """Every word the review pane shows or would show: labels, button text, tooltips, the
    tiles' captions, notes and hover text."""
    pane, montage = tab.pane, tab.pane.montage
    parts = []
    for widget in pane.findChildren(QLabel) + pane.findChildren(QAbstractButton):
        parts += [widget.text(), widget.toolTip()]
    for index, clip in enumerate(montage.clips()):
        parts += [clip.caption, clip.note or "", clip.tag or ""]
        if montage._hover_text is not None:
            parts.append(montage._hover_text(index))
    return "\n".join(parts)


def _scores(run, rid: str = "s1") -> set[str]:
    return {
        f"{row.score:.{digits}f}"
        for row in run._audit_pool("q01")
        if row.recording_id == rid
        for digits in (2, 3)
    }


def test_the_review_pane_shows_no_score_anywhere(qapp, searched):
    """Guard (§20 *Still blind*): no label, button, tooltip, caption, note or hover text in the
    pane carries a candidate's score. The negative case: a caption carrying one is found, so the
    scan can fail."""
    exp, run = searched
    tab = _tab(qapp, exp, run)
    _open(qapp, tab)
    shown = _pane_text(tab)
    leaked = sorted(s for s in _scores(run) if s in shown)
    assert not leaked, f"the review pane shows scores {leaked}"

    montage = tab.pane.montage
    clips = list(montage.clips())
    row = next(r for r in run._audit_pool("q01") if r.recording_id == "s1")
    clips[0] = dataclasses.replace(
        clips[0], caption=f"{clips[0].caption} {row.score:.2f}"
    )
    montage.set_clips(clips, keep_scroll=True)
    assert any(
        s in _pane_text(tab) for s in _scores(run)
    ), "the scan cannot see a score"


def test_remove_then_finish_writes_the_census_and_shows_the_result(qapp, searched):
    exp, run = searched
    tab = _tab(qapp, exp, run)
    _open(qapp, tab)
    pane = tab.pane
    assert pane.finish.isEnabled()
    pane.montage.clip_clicked.emit(0, False)
    assert _drain(qapp, lambda: not tab.writes_pending())
    assert len(run.taken_out("q01")) == 1
    assert pane.undo.isEnabled() and "Removed 1" in pane._undo_line

    pane.finish.click()
    assert _drain(qapp, lambda: not tab.writes_pending() and pane.is_finished())
    drain_busy_calls(qapp)
    (row,) = [r for r in run.review_by_recording("q01") if r["recording_id"] == "s1"]
    assert row["exact"] and row["removed"] == 1
    assert pane.precision.text() == f"{round(100 * 5 / 6)}%"
    assert pane.next.text() == "Next recording: s2"
    assert [title for title, _n in pane.montage.sections()] == [
        "Removed (1)",
        "Kept as rearing (5)",
    ]


def test_a_snapshot_reads_the_review_and_writes_nothing(qapp, searched):
    exp, run = searched
    view = run.view(run.pin(reason="finish"))
    tab = _tab(qapp, exp, view)
    _open(qapp, tab)
    assert tab._reason() == FROZEN
    assert not tab.pane.finish.isEnabled() and tab.pane.finish.toolTip() == FROZEN
    tab.pane.montage.clip_clicked.emit(0, False)
    assert not tab.writes_pending() and not run.taken_out("q01")


# --- nothing to review: one page, one thing to do -------------------------------------------
#
# **The class.** An empty review drawn as a review: a message over an empty list, an empty pane
# and a greyed *Finish review* — four widgets describing their own absence — and, on a run
# finished with no cutoff, "set a cutoff" and "reopen it" at once. With no cutoff the page is a
# headline, a line and one button whose signal the workspace connects, or no button where there
# is nothing to press (a snapshot). Its button is shown only once that signal has a receiver.


@pytest.fixture(scope="module")
def _uncut_template(tmp_path_factory):
    """One searched run with no cutoff, built once and copied per test."""
    exp, _run = _searched(tmp_path_factory.mktemp("uncut"))
    return exp.dir


@pytest.fixture
def uncut(_uncut_template, tmp_path):
    import shutil

    from detail.api.experiment import Experiment

    copy = tmp_path / _uncut_template.name
    shutil.copytree(_uncut_template, copy)
    exp = Experiment(copy)
    return exp, exp.query_run("an_01")


def _empty_tab(qapp, exp, run, *, wired: bool = True):
    """The tab on q01, every press it makes recorded — wired the way the workspace wires it."""
    tab = QueryReviewTab(exp, run)
    pressed: list[str] = []
    if wired:
        tab.go_to_calibrate.connect(lambda: pressed.append("calibrate"))
        tab.reopen_requested.connect(lambda: pressed.append("reopen"))
    tab.resize(1440, 847)
    tab.show()
    tab.set_behavior("q01")
    assert _drain(
        qapp, lambda: tab._pages.currentIndex() == tab.PAGE_EMPTY
    ), "the empty page never showed"
    qapp.processEvents()
    return tab, pressed


def _buttons(tab) -> list:
    return [tab._go_to_calibrate, tab._reopen]


def _shown(tab):
    """The one button the empty page offers, or ``None``; every other one is also disabled,
    which is what a walk over the controls live on a snapshot reads (it skips nothing hidden).
    """
    shown = [b for b in _buttons(tab) if not b.isHidden()]
    assert len(shown) <= 1, [b.text() for b in shown]
    for button in _buttons(tab):
        if button not in shown:
            assert not button.isEnabled(), f"{button.text()!r} hidden but live"
    return shown[0] if shown else None


def _empty_page(tab) -> tuple[str, str, str | None]:
    """``(headline, line, button text or None)`` as shown."""
    button = _shown(tab)
    return (
        tab._empty_title.text(),
        tab._empty_body.text(),
        None if button is None else button.text(),
    )


def test_a_live_run_with_no_cutoff_shows_one_page_and_go_to_calibrate(qapp, uncut):
    exp, run = uncut
    tab, pressed = _empty_tab(qapp, exp, run)
    assert _empty_page(tab) == (
        "rearing has no cutoff yet",
        "The review covers every clip the cutoff keeps, so the cutoff comes first.",
        "Go to Calibrate",
    )
    assert not tab.pane.isVisible() and not tab._list.isVisible(), "an empty review"
    assert _shown(tab).isVisible() and _shown(tab).isEnabled()
    _shown(tab).click()
    assert pressed == ["calibrate"]


def test_a_finished_run_with_no_cutoff_offers_reopen_and_nothing_else(qapp, uncut):
    """A run finished before Finish needed a cutoff: one instruction, Reopen."""
    exp, run = uncut
    analysis_store.set_analysis_status(run._run_dir(), "complete")
    tab, pressed = _empty_tab(qapp, exp, run)
    assert _empty_page(tab) == (
        "This run is finished, and rearing has no cutoff",
        "Reopen the run to set a cutoff on Calibrate, then review its clips here.",
        "Reopen",
    )
    assert not tab.pane.isVisible()
    _shown(tab).click()
    assert pressed == ["reopen"]


def test_a_snapshot_with_no_cutoff_says_so_and_offers_no_button(qapp, uncut):
    """Wired like the others, so the missing button is the snapshot's, not the wiring's."""
    exp, run = uncut
    view = run.view(run.pin(reason="export"))
    tab, pressed = _empty_tab(qapp, exp, view)
    assert _empty_page(tab) == (
        "This is a snapshot, and rearing had no cutoff when it was pinned",
        "Go back to the live run to set a cutoff, then review its clips there.",
        None,
    )
    assert not tab.pane.isVisible() and pressed == []


def test_the_button_shows_only_once_the_workspace_connects_its_signal(qapp, uncut):
    """A button is a claim that pressing it does something: unconnected, it is not shown."""
    exp, run = uncut
    tab, _pressed = _empty_tab(qapp, exp, run, wired=False)
    assert _empty_page(tab)[2] is None
    assert tab._empty_title.text() == "rearing has no cutoff yet"


def test_setting_a_cutoff_brings_the_review_back_and_clearing_it_the_page(qapp, uncut):
    exp, run = uncut
    tab, _pressed = _empty_tab(qapp, exp, run)
    run.set_cutoff("q01", _cutoff_keeping(run, 6))
    tab.reload()
    assert _drain(
        qapp,
        lambda: tab._pages.currentIndex() == tab.PAGE_REVIEW
        and tab._list.topLevelItemCount() == 2,
    ), "the review never came back"
    qapp.processEvents()
    assert tab.pane.isVisible() and tab._list.isVisible()
    assert not tab._empty_title.isVisible()
    assert tab.figure.text().startswith("rearing")
    assert tab._reason() == ""
    assert not any(
        b.isEnabled() for b in _buttons(tab)
    ), "a button live under the review"

    run.set_cutoff("q01", None)
    tab.reload()
    assert _drain(qapp, lambda: tab._pages.currentIndex() == tab.PAGE_EMPTY)
    assert tab._list.topLevelItemCount() == 0 and tab._open is None


def test_calibrate_says_which_of_its_clips_the_review_removed():
    """The strip is the method's own ranked list, so a removed clip stays in it — saying so,
    since it is no longer one of the behavior's instances."""
    import numpy as np

    from detail.core.dataclasses import QueryCandidate
    from detail.gui.modules.analysis_query import _clip_specs

    t_ns = np.arange(10, dtype=np.int64) * 1_000_000
    candidates = [QueryCandidate(1, 3, 0.9, 0.9, 0), QueryCandidate(5, 7, 0.4, 0.4, 0)]
    specs = _clip_specs(
        candidates,
        "s1",
        0,
        "emb_01",
        (np.arange(10), t_ns),
        name="s1",
        removed=frozenset({("s1", 0, 5_000_000, 7_000_000)}),
    )
    assert [s.label for s in specs] == ["s1 0.90", "s1 0.40 · removed"]


def test_moving_the_cutoff_reads_the_open_recording_again(qapp, searched):
    """A recording finished at one cutoff and the cutoff lowered: the pane showed the old clips
    under the new count, still *✓ Reviewed* at a figure that is not one at this cutoff. It reads
    the recording's clips again, goes back to reviewing, and says how far down it was reviewed.
    """
    exp, run = searched
    tab = _tab(qapp, exp, run)
    _open(qapp, tab)
    tab.pane.finish.click()
    assert _drain(qapp, lambda: not tab.writes_pending() and tab.pane.is_finished())
    drain_busy_calls(qapp)

    run.set_cutoff("q01", _cutoff_keeping(run, 9))
    tab.reload()
    assert _drain(
        qapp, lambda: tab._open is not None and len(tab._open["clips"].rows) == 9
    ), "the open recording kept the old cutoff's clips"
    assert not tab.pane.is_finished(), "a review at the old cutoff shown as finished"
    assert "reviewed to" in tab.pane.was.text()
    look = tab._look(tab._row("s1"))
    assert look["line"].startswith("reviewed to") and look["line"].endswith(" new")


def test_calibrate_counts_the_clips_the_review_removed(qapp, tmp_path):
    """CALIBRATE counts the search's matches at a cutoff; a removed clip is still one of them
    but not one of the behavior's instances, so the header says how many."""
    from tests.gui.test_query_workspace import _workspace

    _exp, run, workspace = _workspace(qapp, tmp_path)
    behavior = workspace._behaviors[0].id
    run.set_cutoff(behavior, 0.0)
    assert _drain(qapp, lambda: workspace._curves)
    calibrate = workspace._calibrate
    calibrate._refresh_header(0.0)
    assert "removed in Audit" not in calibrate._header.text()
    rid = run.state().search_recording_ids[0]
    run.take_out_clips(run.review_clips(behavior, rid)[:2], behavior)
    calibrate._refresh_header(0.0)  # the clips were removed at cutoff 0.0
    assert "2 of them removed in Audit" in calibrate._header.text()
    calibrate._refresh_header(1.0)
    assert (
        "removed in Audit" not in calibrate._header.text()
    ), "counted below the cutoff"


def test_an_earlier_one_by_one_audit_stays_in_the_figures_tooltip(qapp, searched):
    """A run audited one clip at a time before the review keeps that figure on screen, apart
    from the review's (§20 *What left*); without one the tooltip says nothing."""
    from tests.audit_seed import draw, judge

    exp, run = searched
    tab = _tab(qapp, exp, run)
    assert tab.figure.toolTip() == ""
    for row in draw(run, label="q01", budget=10):
        judge(run, row.instance_id, True)
    tab.reload()
    assert _drain(
        qapp, lambda: "Earlier one-by-one audit: precision" in tab.figure.toolTip()
    )


def test_a_reviewed_recordings_precision_is_its_rows_headline(qapp, tmp_path):
    """The query AUDIT's rows draw as the clustering ones do: a recording reviewed exactly at
    the cutoff carries its figure as ``figure`` (the delegate's headline), its line says how many
    were removed; a recording not reviewed carries no figure."""
    from detail.gui.query_review import QueryReviewTab

    row = {
        "recording_id": "s1",
        "name": "s1",
        "size": 10,
        "exact": True,
        "estimate": {"precision": 0.9},
        "at_floor": None,
    }
    tab = QueryReviewTab.__new__(QueryReviewTab)
    tab._open = None
    tab._data = {"taken": {}}
    tab._reopened = set()
    look = QueryReviewTab._look(tab, row)
    assert look["figure"] == "90%" and look["line"] == "0 removed"
    look = QueryReviewTab._look(tab, dict(row, exact=False))
    assert look["figure"] == ""
