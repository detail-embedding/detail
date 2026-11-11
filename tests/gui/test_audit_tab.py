"""The §19 audit tab on a query run — the surface, not the arithmetic.

Clustering's audit is `gui/label_check.py`, tested in `tests/gui/test_label_check.py`.

`tests/storage/test_behavior_audit.py` covers the estimator and `tests/api/test_query.py` the
API. What is left is the part only a widget can get wrong, and every item is a failure that
produces *plausible output* rather than an error:

- **The score leaking into a blind review.** Seeing 0.95 pulls a judgment toward accepting and
  inflates precision by exactly the quantity the audit measures. Nothing raises; the number is
  simply optimistic and looks like any other number.
- **A judgment recorded by accident.** The keys make a 25-clip pass fast; a window-scoped
  shortcut would make `Y` pressed on another tab record a human decision about a clip nobody is
  looking at. That is the same defect as the leak, arriving from the other direction.
- **A number rendered from too few decisions.** `1.00` from n=3 and `1.00` from n=50 are the
  same pixels and different claims.
- **A screen that spreads until the work is no longer on it.** This tab puts a list and a player
  in one splitter, and a splitter hands every child its whole height: the list drew as many rows
  as the window was tall and the block holding the clip grew with it, so the three buttons that
  judge the clip ended up at the bottom edge of the screen. Nothing fails — every widget is
  present, correct and reachable by scrolling. The class is `f9350f0a`'s, found here at its third
  site (`docs/recorded_classes.md`), and the measurements are in the section that guards it.
- **An id where a name belongs (§11).** The clip rows, the line under the video and the
  by-recording table each printed `recording_id`. An id is a plausible thing to print, which is
  why none of this raised.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QDialog, QLabel  # noqa: E402

from tests.gui.layout_checks import assert_top_justified

from detail.gui import audit_tab as audit_tab_module  # noqa: E402
from detail.gui.sizing import lines  # noqa: E402
from detail.gui.audit_tab import (  # noqa: E402
    BATCH,
    SCOPES,
    AuditTab,
    format_precision,
)
from detail.storage import behavior_audit as audit  # noqa: E402

from tests.api.test_query import _searched  # noqa: E402
from tests.gui.conftest import drain_busy_calls, drain_thread_pool  # noqa: E402
from tests.gui.test_query_workspace import _drain  # noqa: E402


def _tab(qapp, tmp_path, cutoff=0.0, **searched):
    exp, run = _searched(tmp_path, **searched)
    run.set_cutoff("q01", cutoff)
    tab = AuditTab(exp, run)
    tab.set_behavior("q01")
    assert _drain(qapp, lambda: bool(tab._headline.text()))
    # ...and then until nothing is still loading. The headline is one widget; the plot panel, the
    # tally, the player and the body each populate off-thread on their own schedule. 20 tests in
    # this file assert something is *absent*, and an unbuilt widget satisfies that for free — so
    # draining on one signal leaves those assertions able to pass against a half-built tab (#533).
    drain_busy_calls(qapp)
    drain_thread_pool(qapp)
    return exp, run, tab


def _labels(widget) -> str:
    return " ".join(w.text() for w in widget.findChildren(QLabel))


# --------------------------------------------------------------------------- #
# blind review, in the page
# --------------------------------------------------------------------------- #
def test_the_first_thing_you_see_says_what_to_do(qapp, tmp_path):
    """The pane used to be blank, which reads as broken rather than as "nothing yet", and the
    only way forward was a button below the fold offering to review 25 *more* when you had
    reviewed none."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    assert "Start auditing" in tab._start.text()
    assert tab._bar.value() == 0
    assert "none reviewed yet" in tab._intro_headline.text()
    # `isHidden`, not `isVisible`: a child of an unshown widget is never "visible", so that
    # assertion would hold whatever the button did.
    assert tab._continue.isHidden(), "offering to continue with nothing left half-done"
    assert tab._list_model.rowCount() == 0


# --------------------------------------------------------------------------- #
# what the tab is before it has anything to show
# --------------------------------------------------------------------------- #
def test_nothing_reviewed_shows_one_action_and_not_the_workspace(qapp, tmp_path):
    """The class: **a screen whose every region describes its own absence.**

    Before an audit there is exactly one useful thing to do, and the three-column workspace
    around it was four regions saying so separately — a clip list with no clips, a prompt adrift
    in a large void, two tables reading "not audited", and a pair of empty axes. It read as a
    page that had failed to load.

    Asserted on the *page*, not on the presence of a label: the old empty state also had a
    sentence in it, and a test that only looked for the sentence passed on both.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)

    assert tab._body.currentIndex() == 0, "the workspace is showing with nothing in it"
    assert "none reviewed yet" in tab._intro_headline.text()
    assert "Start auditing" in tab._intro_start.text()
    assert tab._intro_start.isEnabled()
    # `isHidden`, not `isVisible`: nobody showed this tab, so every child reports invisible
    # and that assertion would hold whether the headline was hidden on purpose or not.
    assert (
        tab._headline.isHidden()
    ), "the run headline repeats what the intro already says"


def test_the_workspace_arrives_with_the_data_it_is_for(qapp, tmp_path):
    """The other half. A mode that never switches back is a mode nobody can leave."""
    _exp, run, tab = _tab(qapp, tmp_path)
    for row in run.audit_sample(label="q01", budget=3):
        run.label_audit(row.instance_id, True)
    tab.reload()
    assert _drain(qapp, lambda: tab._body.currentIndex() == 1)
    assert (
        not tab._headline.isHidden()
    ), "the headline stayed hidden with a workspace to label"


def test_pressing_start_on_the_intro_opens_the_workspace(qapp, tmp_path):
    """Through the button, which is the only way a user gets out of the intro.

    This is the one that was missing, and the bug it now covers shipped because of it: the
    sibling tests drove `reload()` after judging instances behind the widget's back, which
    reaches the mode switch by a different road. `_start_batch` ends at `_show_current`, which
    switched the *review pane's* stack and not the page's — so the batch was drawn, the clip was
    loaded, and pressing "Start auditing" looked like it did nothing at all.

    Constructing a widget proves it builds; driving the control proves it is wired.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    assert tab._body.currentIndex() == 0, "the fixture is not on the intro"

    tab._intro_start.click()
    qapp.processEvents()

    assert (
        tab._body.currentIndex() == 1
    ), "pressing Start left the page on the intro — the batch is drawn and invisible"
    assert tab._queue, "no batch was drawn at all"
    assert tab._stack.currentIndex() == 1, "the review pane is still showing its prompt"


def test_an_empty_scope_after_auditing_keeps_the_workspace(qapp, tmp_path):
    """Filtering to a scope with no rows is an empty *view*, not an empty tab.

    The distinction the mode has to make: bouncing back to "start auditing" because the
    `unclear` filter is empty would throw away the audit you are in the middle of reading.
    """
    _exp, run, tab = _tab(qapp, tmp_path)
    for row in run.audit_sample(label="q01", budget=3):
        run.label_audit(row.instance_id, True)
    tab.reload()
    assert _drain(qapp, lambda: tab._body.currentIndex() == 1)

    tab._scopes["unclear"].setChecked(True)  # nothing was marked unclear
    qapp.processEvents()
    assert (
        tab._body.currentIndex() == 1
    ), "an empty filter sent the tab back to its intro"


def test_the_intro_start_button_is_dead_until_the_run_has_been_read(qapp, tmp_path):
    """A control that is live before anything has been read is live on a snapshot too — and the
    walk that catches that keys on the button's *label*, so an unlabelled one slips past it
    (`tests/gui/test_query_workspace.py`)."""
    from detail.gui.audit_tab import AuditTab

    exp, run = _searched(tmp_path)
    tab = AuditTab(exp, run)  # constructed, nothing read yet
    assert not tab._intro_start.isEnabled()
    assert tab._intro_start.text() == ""
    # The rest of the writers too: whether the run is finished is read with everything else, so
    # before that read lands nothing here knows whether the run would accept a decision.
    assert not tab._start.isEnabled()
    assert not any(button.isEnabled() for button in tab._buttons)


def test_a_finished_runs_audit_is_dead_and_says_reopen(qapp, tmp_path):
    """`_record` refuses a finished run, and this tab asked only `editable` — which a finished
    run is — so Start stayed live and drew into a `ValueError`. Dead now, with the reason that is
    undone by Reopen rather than the snapshot's, and reaching past the button writes nothing.
    """
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    run.finalize()
    tab = AuditTab(exp, run)
    tab.set_behavior("q01")
    assert _drain(qapp, lambda: tab._finished is not None)
    assert "nothing to audit" not in tab._headline.text(), "the fixture must have clips"

    assert not tab._start.isEnabled()
    assert not tab._intro_start.isEnabled()
    assert tab._start.toolTip() == AuditTab.FINISHED
    before = len(audit.read_audit(run._audit_dir()))
    tab._start_batch()
    assert not tab._batch
    assert len(audit.read_audit(run._audit_dir())) == before


def test_the_button_offers_more_once_you_have_started(qapp, tmp_path):
    _exp, run, tab = _tab(qapp, tmp_path)
    for row in run.audit_sample(label="q01", budget=3):
        run.label_audit(row.instance_id, True)
    tab.reload()
    assert _drain(qapp, lambda: "more" in tab._start.text())
    assert "Start auditing" not in tab._start.text()


def test_the_total_audited_is_always_on_screen(qapp, tmp_path):
    """Per-batch progress resets each time, so the running total lives in the headline."""
    _exp, run, tab = _tab(qapp, tmp_path)
    for row in run.audit_sample(label="q01", budget=3):
        run.label_audit(row.instance_id, True)
    tab.reload()
    # Wait for the value, not the word — "audited" is in the headline before the reload lands,
    # so draining on it asserts against stale text.
    assert _drain(qapp, lambda: "3 of " in tab._headline.text())


def _audited(text: str) -> tuple[int, int, int]:
    """``(N, M, below)`` from the headline's "N of M instances audited · k earlier …"."""
    import re

    n, m = re.search(r"(\d+) of (\d+) instances audited", text).groups()
    below = re.search(r"(\d+) earlier decisions? (?:is|are) on", text)
    return int(n), int(m), int(below.group(1)) if below else 0


def test_raising_the_cutoff_cannot_push_the_audited_count_past_its_total(
    qapp, tmp_path
):
    """The count and its total have to come from one set. N was every decision in the log and
    M only what the current cutoff keeps, so raising the cutoff read "75 of 68" (#629). The
    decisions the cutoff now excludes are named beside it rather than dropped from view.
    """
    _exp, run, tab = _tab(qapp, tmp_path)
    _finish_batch(qapp, tab)
    drain_busy_calls(qapp)
    assert "earlier decision" not in tab._headline.text()
    judged = {r.instance_id for r in run.audit_decisions() if r.human_label is not None}

    # 0.9 keeps 2 of this fixture's 30 instances (see the below-the-cutoff list test).
    run.set_cutoff("q01", 0.9)
    kept = {r.instance_id for r in run.artifact().instances if r.source_label == "q01"}
    assert len(judged - kept) > 0, "the cutoff excluded nothing that was judged"
    tab.reload()
    assert _drain(qapp, lambda: "earlier decision" in tab._headline.text())

    n, m, below = _audited(tab._headline.text())
    assert n <= m
    assert (n, m, below) == (len(judged & kept), len(kept), len(judged - kept))


def test_another_behaviors_decisions_do_not_move_this_ones_count(qapp, tmp_path):
    """Both behaviors here are planted on the same span, so their clips overlap exactly — and
    `instance_id` hashes the behavior, so a decision about grooming is never one about rearing.
    """
    from tests.api.test_query import _two_behaviors

    exp, run = _searched(tmp_path, behaviors=_two_behaviors(["s1", "s2"]))
    run.set_cutoff("q01", 0.0)
    run.set_cutoff("q02", 0.0)
    for row in run.audit_sample(label="q01", budget=3):
        run.label_audit(row.instance_id, True)
    tab = AuditTab(exp, run)
    tab.set_behavior("q01")
    assert _drain(qapp, lambda: "3 of " in tab._headline.text())
    # Every read in flight delivered first, or the second headline below is the constructor's
    # first read landing late — unchanged whatever the count does, so the test cannot fail.
    drain_busy_calls(qapp)
    before = _audited(tab._headline.text())

    for row in run.audit_sample(label="q02", budget=5):
        run.label_audit(row.instance_id, False)
    tab._headline.setText("")
    tab.reload()
    assert _drain(qapp, lambda: bool(tab._headline.text()))
    drain_busy_calls(qapp)
    assert _audited(tab._headline.text()) == before == (3, before[1], 0)


def test_the_batch_bar_tracks_the_clips_judged(qapp, tmp_path):
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    assert tab._bar.maximum() == len(tab._queue)
    assert tab._bar.value() == 0
    tab._decide(True)
    assert tab._bar.value() == 1


def test_the_audit_player_opens_with_a_second_of_context(qapp, tmp_path):
    """Judging a clip is judging what led into it; this page passes 1 s, other pages pass none
    (issue 627). Asked of the player, since a page default lives in the call, not the class.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    assert tab._player.context_s() == 1.0


def test_the_player_shows_the_video_not_the_input_image(qapp, tmp_path):
    """One pane, and it is the footage with the subject marked.

    The second pane was the *input image* — the constructor's rendering of what the embedder
    saw — which answers a question about the method rather than about the animal, and took half
    the space the video had.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    assert tab._player._masked is True
    # The property, not one foreign attribute name: `_ClipPlayer` never defined `_sil` at
    # all (it belongs to a different widget), so that disjunction was unconditionally true.
    from detail.gui.image_view import ImageView

    assert len(tab._player.findChildren(ImageView)) == 1, "a second pane is back"


def test_the_review_is_not_a_dialog(qapp, tmp_path):
    """The popup hid the statistics behind the thing that changes them, and never said which
    behavior was being judged. This is the regression test for it coming back."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    assert not tab.findChildren(QDialog), "the review is in a dialog again"


def _renderings(value: float) -> set[str]:
    """Every way a float plausibly reaches a label.

    The guard used to look for `f"{score:.3f}"` and nothing else, so a leak printed to two
    decimals walked straight past it — verified by injecting one onto the pre-decision pane and
    watching all 55 tests pass. A leak inflates precision by exactly the quantity the audit
    measures, so this covers the ways a leak actually gets written, not the one formatting the
    code happens to use today.

    One decimal is excluded deliberately: the pane renders a time and a duration at 1dp
    ("0.6 s"), so a 1dp score is indistinguishable from them by text alone, and a guard that
    false-alarms gets loosened until it says nothing.
    """
    return {f"{value:.{dp}f}" for dp in (2, 3, 4)} | {str(value), repr(value)}


def _assert_no_score_visible(tab) -> None:
    """No queued clip's score appears anywhere the reviewer reads before deciding."""
    scored = [row for row in tab._queue if row.score is not None]
    assert scored, "the fixture must produce scored instances or this proves nothing"
    # The *review* surfaces only. The stats panel beside them legitimately renders rates — a
    # precision of 1.00 must not be read as a leaked score of 1.0 — so widening this to the
    # whole tab would make it false-alarm.
    # `_last` is excluded deliberately: it is the *reveal*, shown after a decision, and
    # governed by its own tests (`test_a_revision_does_not_re_reveal_the_score`). Everything
    # here is what the reviewer reads while still deciding.
    panes = [tab._progress, tab._where, tab._question, tab._prompt]
    listed = " ".join(
        tab._list_model.item(r, c).text()
        for r in range(tab._list_model.rowCount())
        for c in range(tab._list_model.columnCount())
        if tab._list_model.item(r, c)
    )
    shown = " ".join(w.text() for w in panes) + " " + listed
    for row in scored:
        for rendering in _renderings(row.score):
            assert (
                rendering not in shown
            ), f"a clip's score is on screen before judging, as {rendering!r}"


def test_the_leak_check_catches_a_leak(qapp, tmp_path):
    """Guard the guard. A score check matching one formatting passes against a real leak
    written in any other — which is exactly how the previous version shipped."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    leaked = next(row for row in tab._queue if row.score is not None)
    tab._where.setText(f"SCORE {leaked.score:.2f} · leaked at two decimals")
    with pytest.raises(AssertionError, match="on screen before judging"):
        _assert_no_score_visible(tab)


def test_no_widget_carries_the_score_before_a_decision(qapp, tmp_path):
    """Asserted as *absence*, not `isVisible() is False`: a hidden label is one
    `setVisible(True)` away from a biased estimate, and a visibility assertion passes on one.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    scored = [row for row in tab._queue if row.score is not None]
    assert scored, "the fixture must produce scored instances or this proves nothing"

    _assert_no_score_visible(tab)

    # ...and it is not handed to the player either. `ClipSpec` carries a `score` and the player
    # ignores it *today*; a caption that started rendering it would reintroduce the bias with
    # nothing here failing, so the intent is pinned rather than left resting on that.
    assert tab._player._clip is not None
    assert tab._player._clip.score is None


def test_the_score_is_revealed_after_the_decision(qapp, tmp_path):
    """Withheld, not withdrawn — the reviewer has already committed by then."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    first = tab._current()
    tab._decide(True)
    assert f"{first.score:.3f}" in _labels(tab)
    assert "you said yes" in _labels(tab)


def test_a_decision_reaches_the_audit_log(qapp, tmp_path):
    _exp, run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    tab._decide(True)
    tab._decide(False)
    judged = [
        r.human_label
        for r in audit.latest_decisions(run._run_dir())
        if r.human_label is not None
    ]
    assert sorted(judged, key=str) == [False, True]


def test_unclear_is_recorded_as_undecided_not_as_a_rejection(qapp, tmp_path):
    """`human_label` is tri-state and readers test `is True`. Folding "unclear" into False
    would count a clip nobody could judge against the method."""
    _exp, run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    target = tab._current().instance_id
    tab._decide(None)
    rows = {
        r.instance_id: r.human_label for r in audit.latest_decisions(run._run_dir())
    }
    assert rows[target] is None


# --------------------------------------------------------------------------- #
# the keys
# --------------------------------------------------------------------------- #
def test_each_judgment_has_a_key_shown_in_its_label(qapp, tmp_path):
    """Every judgment's key is shown in its label, so the shortcut is discoverable.

    The ampersand half of this moved to `detail/gui/requirements.py`, where it runs over all
    ~126 buttons in the app rather than the three here — which is the whole reason it moved. What
    stays is the part specific to this tab: that the key each shortcut binds is written on the
    button that performs it.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    keys = {s.key().toString().upper() for s in tab._shortcuts} - {"LEFT", "RIGHT"}
    # Home row under the left hand, so the right can stay on the mouse. `Y`/`N`/`U` read
    # better and type worse — three separate reaches, 25 times.
    assert keys == {"S", "D", "F"}
    shown = " ".join(b.text().upper() for b in tab._buttons)
    for key in keys:
        assert f"({key})" in shown, f"{key} is a shortcut nothing on screen mentions"


def test_a_shortcut_cannot_fire_from_another_tab(qapp, tmp_path):
    """The worst thing this surface can do is record a judgment nobody made.

    A bare shortcut defaults to `WindowShortcut`, so `Y` pressed while Calibrate is showing —
    or typed into a filter box — would judge whatever clip happened to be queued. The context
    is what stops it, and nothing about the wrong context looks wrong on screen.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    for shortcut in tab._shortcuts:
        assert (
            shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
        ), "a judgment or navigation key is live across the whole window"


def test_a_shortcut_does_nothing_while_the_buttons_are_disabled(qapp, tmp_path):
    """`_decide` guards on the buttons being enabled, and that clause was unreachable in the
    test that documented it: with nothing judged the queue is empty, so it returned one line
    earlier at `instance is None`. Reach the state the clause exists for — a clip queued, the
    buttons disabled — and assert nothing is recorded."""
    _exp, run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    assert tab._current() is not None
    before = len(audit.read_audit(run._run_dir()))
    tab._set_buttons(False)
    tab._decide(True)
    assert len(audit.read_audit(run._run_dir())) == before, "a disabled button recorded"


def test_the_buttons_are_disabled_when_there_is_nothing_to_judge(qapp, tmp_path):
    """The key is only half the guard: a live shortcut with no clip queued must do nothing."""
    _exp, run, tab = _tab(qapp, tmp_path)
    assert not any(b.isEnabled() for b in tab._buttons)
    before = len(audit.read_audit(run._run_dir()))
    tab._decide(True)  # what the key would reach
    assert len(audit.read_audit(run._run_dir())) == before


# --------------------------------------------------------------------------- #
# the tables
# --------------------------------------------------------------------------- #
def test_the_by_label_table_lists_every_behavior_and_navigates(qapp, tmp_path):
    """It is the answer to "what should I audit next", so it has to show behaviors the current
    scope is not on — and clicking one has to move there."""
    from tests.api.test_query import _two_behaviors

    exp, run = _searched(tmp_path, behaviors=_two_behaviors(["s1", "s2"]))
    run.set_cutoff("q01", 0.0)
    run.set_cutoff("q02", 0.0)
    tab = AuditTab(exp, run)
    tab.set_behavior("q01")
    assert _drain(qapp, lambda: tab._by_label_model.rowCount() > 1)

    labels = [
        tab._by_label_model.item(i, 0).data(Qt.ItemDataRole.UserRole)
        for i in range(tab._by_label_model.rowCount())
    ]
    assert set(labels) == {"q01", "q02"}

    asked: list = []
    tab.behavior_requested.connect(asked.append)
    row = labels.index("q02")
    tab._on_label_row(tab._by_label_model.index(row, 0))
    assert asked == ["q02"], "clicking a row did not ask the workspace to switch"
    assert tab._label == "q02"


def test_switching_behavior_mid_batch_loses_nothing(qapp, tmp_path):
    """Nothing is discarded, so nothing has to warn: the unjudged remainder returns to the pool
    and is offered first next time (`AuthoredRun.audit_sample`)."""
    from tests.api.test_query import _two_behaviors

    exp, run = _searched(tmp_path, behaviors=_two_behaviors(["s1", "s2"]))
    run.set_cutoff("q01", 0.0)
    tab = AuditTab(exp, run)
    tab.set_behavior("q01")
    assert _drain(qapp, lambda: bool(tab._headline.text()))
    tab._start_batch()
    abandoned = {row.instance_id for row in tab._queue}

    tab.set_behavior("q02")
    assert tab._queue == [], "the batch carried across a behavior switch"
    tab.set_behavior("q01")
    assert _drain(qapp, lambda: bool(tab._headline.text()))
    tab._start_batch()
    assert abandoned <= {row.instance_id for row in tab._queue}


# --------------------------------------------------------------------------- #
# what it renders
# --------------------------------------------------------------------------- #
def test_an_unaudited_run_says_so_rather_than_showing_a_rate(qapp, tmp_path):
    _exp, _run, tab = _tab(qapp, tmp_path)
    assert "not audited" in tab._headline.text()


def test_a_rate_from_too_few_decisions_is_words_not_a_number(qapp, tmp_path):
    """The rendered string, not the dict — storage tests the dict. This is the half a surface
    gets wrong on its own by formatting `precision` regardless of `below_min_n`."""
    _exp, run, tab = _tab(qapp, tmp_path)
    for row in run.audit_sample(label="q01", budget=3):
        run.label_audit(row.instance_id, True)
    tab.reload()
    assert _drain(qapp, lambda: "too few" in tab._headline.text())
    assert "1.00" not in tab._headline.text()


def test_enough_decisions_render_the_rate_with_its_interval(qapp, tmp_path):
    _exp, run, tab = _tab(qapp, tmp_path)
    for row in run.audit_sample(label="q01", budget=audit.MIN_N_FOR_ESTIMATE + 4):
        run.label_audit(row.instance_id, True)
    tab.reload()
    assert _drain(qapp, lambda: "n=" in tab._headline.text())
    text = tab._headline.text()
    assert "1.00" in text and "–" in text and "too few" not in text


def test_format_precision_distinguishes_not_audited_from_not_enough():
    """Two states a single "no number" would collapse: work outstanding, versus work done that
    does not support a claim."""
    assert format_precision(None) == "not audited"
    assert format_precision({"n": 0}) == "not audited"
    assert "too few" in format_precision(
        {"n": 4, "below_min_n": True, "precision": 1.0, "interval": (0.5, 1.0)}
    )


def test_the_confidence_panel_is_absent_until_there_is_something_to_plot(
    qapp, tmp_path
):
    """Absent, not flat at zero.

    A curve drawn from nothing reads as "precision is low" rather than "nobody has looked" —
    the same mistake the headline avoids by saying "not audited" instead of `0.00`. It also has
    its own 0–1 axis here, which the Calibrate overlay it replaces never did: that was drawn as
    `precision × max_instances` on the *instances* axis, so no value could be read off it.
    """
    _exp, run, tab = _tab(qapp, tmp_path)
    assert not tab._plot.isVisibleTo(tab), "a curve was drawn before any judgment"

    for row in run.audit_sample(label="q01", budget=audit.MIN_N_FOR_ESTIMATE + 4):
        run.label_audit(row.instance_id, True)
    tab.reload()
    assert _drain(qapp, lambda: tab._plot.isVisibleTo(tab))
    assert tab._curve_heading.isVisibleTo(tab)
    # the axis is precision's own, not a rescaling onto some other quantity
    # No y-range assertion here. A pyqtgraph ViewBox *defaults* to 0-1 on both axes, so on a
    # plot whose auto-range was never enabled the check holds with `setYRange` removed, with
    # `bound_axes` removed, and with both removed — verified. The precision axis's domain is
    # guarded where the mechanism lives (`tests/gui/test_plot_bounds.py`), and the x-range test
    # below is sensitive only because a resize genuinely crops it.


# --------------------------------------------------------------------------- #
# going back to fix a mistake
# --------------------------------------------------------------------------- #
def test_you_can_go_back_and_change_an_answer(qapp, tmp_path):
    """A misclick on clip 3 of 25 should not mean living with it. The log appends, so a
    revision supersedes without destroying what was said first."""
    _exp, run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    first = tab._current()
    tab._decide(True)
    assert tab._index == 1, "a first answer advances"

    tab._step(-1)
    assert tab._current().instance_id == first.instance_id
    tab._decide(False)
    assert tab._index == 0, "a revision stays put so you can see it took"

    rows = [
        r
        for r in audit.read_audit(run._run_dir())
        if r.instance_id == first.instance_id
    ]
    assert len(rows) >= 2, "the earlier answer was overwritten rather than superseded"
    latest = {
        r.instance_id: r.human_label for r in audit.latest_decisions(run._run_dir())
    }
    assert latest[first.instance_id] is False


def test_a_revision_does_not_re_reveal_the_score(qapp, tmp_path):
    """The score is shown once, after the first judgment. Re-showing it while someone
    reconsiders is exactly the bias the blind review removes — and correcting a misclick does
    not need the number."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    first = tab._current()
    tab._decide(True)
    tab._step(-1)
    tab._decide(False)

    assert "changed to no" in tab._last.text()
    assert f"{first.score:.3f}" not in tab._last.text()


def test_a_revisited_clip_shows_what_you_said(qapp, tmp_path):
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    tab._decide(True)
    tab._step(-1)
    assert "you said yes" in tab._progress.text()


def test_navigation_stops_at_the_ends_rather_than_wrapping(qapp, tmp_path):
    """Wrapping from the last clip to the first reads as a reset, and a reviewer who has lost
    their place cannot tell the two apart."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    tab._step(-1)
    assert tab._index == 0 and not tab._back.isEnabled()
    for _ in range(len(tab._queue) + 5):
        tab._step(1)
    assert tab._index == len(tab._queue) - 1 and not tab._forward.isEnabled()


def test_the_draw_buttons_sit_with_the_list_they_fill(qapp, tmp_path):
    """One column answers "what am I looking at" — the walks, and how to get more of them.
    With the buttons on a summary page you had to leave the clip to reach them."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    sidebar = tab._list.parentWidget()
    assert tab._start in sidebar.findChildren(type(tab._start))
    assert tab._continue in sidebar.findChildren(type(tab._continue))
    assert tab._stack.currentIndex() == 0, "nothing judged yet, so nothing to show"
    tab._start_batch()
    assert tab._stack.currentIndex() == 1, "the player did not replace the prompt"


def test_the_stats_pane_packs_at_the_top(qapp, tmp_path):
    """`gui_design` §3, applied to a pane rather than a page. With nothing in the column
    declaring that it wants the slack, Qt spread it across the headings: "BY BEHAVIOR" was
    laid out at 324px in a 756px pane, three hundred pixels above its own numbers."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    assert_top_justified(tab)


def test_the_summary_tables_fill_their_column(qapp, tmp_path):
    """The other axis of the same rule — and this one went the wrong way first.

    Capping them to their content (`fit_table_width`) is right when the leftover would be dead
    grid *inside* a table. These two sit alone in a 400px column, so the cap put ~200px of
    content in a ~200px table and left the rest of the column blank beside them, with the
    confidence plot underneath filling the whole width. Reported as "the tables don't fill the
    pane". The same cap produced the opposite symptom on a clustering run, where
    `Cluster | Name | precision` overflowed the column and the table grew a horizontal
    scrollbar rather than compressing.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab.resize(1600, 1000)
    tab.show()
    qapp.processEvents()
    try:
        for view in (tab._by_label, tab._by_recording):
            assert (
                view.horizontalHeader().stretchLastSection()
            ), f"{view.objectName() or view} neither fills its column nor hugs its rows"
            pane = view.parentWidget()
            assert (
                view.width() >= pane.width() - 2 * view.x() - 2
            ), f"the table is {view.width()}px in a {pane.width()}px column"
            assert (
                view.horizontalScrollBar().maximum() == 0
            ), "the columns overflowed instead of compressing"
    finally:
        tab.hide()


def test_the_summary_tables_hug_their_rows(qapp, tmp_path):
    """`gui_design` §3: a list table caps its height to its content. Stretched, one behavior
    and one recording rendered as two rows above ~700px of empty grid, with the confidence
    panel pushed off the bottom."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    for view in (tab._by_label, tab._by_recording):
        rows = view.model().rowCount()
        header = view.horizontalHeader().sizeHint().height()
        expected = header + rows * view.verticalHeader().defaultSectionSize()
        assert view.maximumHeight() <= expected + 2 * view.frameWidth()


# --------------------------------------------------------------------------- #
# one walk, chosen in the sidebar
# --------------------------------------------------------------------------- #
def _finish_batch(qapp, tab, answers=None):
    """Answer every clip in the drawn batch, returning what was said."""
    before = len(tab._judged)
    tab._start_batch()
    said = []
    for i in range(len(tab._queue)):
        value = answers[i % len(answers)] if answers else True
        said.append(value)
        tab._decide(value)
    # On the *recompute*, not on the tally: the tally is set synchronously inside `_decide`
    # while the panels and the judged list are still being read off the UI thread, so draining
    # on it hands back a tab whose sidebar is one batch out of date.
    assert _drain(qapp, lambda: len(tab._judged) > before), "the batch never reported"
    assert tab._tally
    return said


def _rows(tab) -> list[tuple]:
    model = tab._list_model
    return [
        tuple(
            model.item(r, c).text()
            for c in range(model.columnCount())
            if model.item(r, c)
        )
        for r in range(model.rowCount())
    ]


def test_finishing_a_batch_says_so_without_taking_the_clip_away(qapp, tmp_path):
    """It used to swap in a summary page you then had to leave. The tally is a line beside
    the clip counter instead: the precision in the heading is the result, this is the sanity
    check on your own pass, and the clip you just judged is still there to correct."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    said = _finish_batch(qapp, tab, answers=[True, False, None])
    assert tab._stack.currentIndex() == 1, "the clip was taken away"
    # The whole string. `"9 yes" in "19 yes"` is True, so containment passed against every
    # count inflated by ten.
    assert tab._tally == (
        f"✓ batch done — {len(said)} reviewed · {said.count(True)} yes · "
        f"{said.count(False)} no · {said.count(None)} unclear"
    )
    assert tab._tally in tab._progress.text()


def test_the_tally_clears_when_the_next_walk_starts(qapp, tmp_path):
    _exp, _run, tab = _tab(qapp, tmp_path)
    _finish_batch(qapp, tab)
    tab._start_batch()
    assert not tab._tally


def test_the_clip_stops_playing_when_the_batch_ends(qapp, tmp_path):
    """A looping clip beside the numbers is motion competing with the result.

    The player must be *playing* first: with no decodable clip `_toggle_play` snaps the button
    straight back, so asserting "not playing" at the end is satisfied by one that never
    started and the guard means nothing.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    for _ in range(len(tab._queue) - 1):
        tab._decide(True)
    # Length and "a frame is up" now live on the scrubber, which owns playback — setting
    # `_len` alone leaves its total at 0, and a player with nothing to play snaps Play straight
    # back. Give it both, which is what a decoded clip does.
    tab._player._len = 10
    tab._player._scrubber.set_total(10)
    tab._player._scrubber._has_rendered = True
    tab._player.play()
    assert tab._player.is_playing(), "the player never started"
    tab._decide(True)
    assert _drain(qapp, lambda: len(tab._judged) > 0)
    assert not tab._player.is_playing()


def test_a_refresh_does_not_restart_the_clip_already_on_screen(qapp, tmp_path):
    """The play state cannot see this. `set_clip` autoplays, so the refresh that follows a
    batch-ending judgment put the loop straight back — but in a fixture whose clips do not
    decode, `_toggle_play` snaps the button off again and "not playing" holds either way.
    A user with real footage saw it loop. So assert the mechanism: the clip already on screen
    is not re-set, which is also what stops a refresh re-decoding a frame nobody asked for.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    _finish_batch(qapp, tab)
    calls = []
    original = tab._player.set_clip
    tab._player.set_clip = lambda spec: (calls.append(spec), original(spec))[1]

    tab._judged = []  # or the drain is satisfied by the previous read
    tab.reload()
    assert _drain(qapp, lambda: bool(tab._judged)), "the refresh never landed"
    assert calls == [], "a refresh re-set the clip that was already showing"

    tab._list.selectRow(1)
    assert len(calls) == 1, "a different clip must still be loaded"


def test_a_skipped_clip_means_the_batch_has_not_reported(qapp, tmp_path):
    """`→` walks past a clip without judging it, so "answered the last one" and "answered
    them all" are different questions, and only the second ends a batch."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    tab._step(1)  # skip clip 1 unjudged
    for _ in range(len(tab._queue) - 1):
        tab._decide(True)
    assert not tab._tally


# --------------------------------------------------------------------------- #
# the sidebar: what to walk, and how to get more of it
# --------------------------------------------------------------------------- #
def test_selecting_a_row_shows_that_clip_and_records_nothing(qapp, tmp_path):
    """A single click says which clip to look at — it is the job `_BrowseTab` does beside its
    own player. Nothing but the three buttons may write to the log."""
    _exp, run, tab = _tab(qapp, tmp_path)
    _finish_batch(qapp, tab)
    tab._scopes["judged"].setChecked(True)
    before = len(audit.read_audit(run._run_dir()))

    expected = tab._scope_rows("judged")[3][0].instance_id
    tab._list.selectRow(3)
    assert tab._index == 3
    # Against the *list*, not `tab._queue[3]`: `_current()` is defined as `_queue[_index]`, so
    # comparing the two was true whatever the click did.
    assert tab._current().instance_id == expected
    assert "you said yes" in tab._progress.text()
    assert (
        len(audit.read_audit(run._run_dir())) == before
    ), "selecting recorded something"


def test_the_walk_and_the_list_follow_each_other(qapp, tmp_path):
    """Both directions, and neither may re-drive the other into a loop."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    _finish_batch(qapp, tab)
    tab._list.selectRow(2)
    tab._step(1)
    assert tab._index == 3
    assert tab._list.currentIndex().row() == 3, "the list did not follow the walk"

    tab._list.selectRow(0)
    assert tab._index == 0, "the walk did not follow the list"


def test_every_judgment_is_reachable_not_just_the_last_batch(qapp, tmp_path):
    """The reason the sidebar exists. `Back to clip 25` reached one clip; three batches in, a
    mistake in the first was unreachable and no surface even showed it."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    _finish_batch(qapp, tab)
    first = {row.instance_id for row in tab._batch}
    _finish_batch(qapp, tab)

    listed = {row.instance_id for row, _said in tab._scope_rows("judged")}
    assert first <= listed, "an earlier batch's judgments dropped off the list"
    assert len(listed) > len(first)


def test_a_judgment_below_the_cutoff_is_still_listed(qapp, tmp_path):
    """`_audit_pool` is cutoff-free so that **one audit serves every cutoff** — `instance_id`
    is a content hash precisely so a decision survives the artifact being rebuilt. Built from
    `artifact().instances` instead, the list silently loses every judgment the current cutoff
    excludes, and the count beside it agrees with itself."""
    _exp, run, tab = _tab(qapp, tmp_path)
    _finish_batch(qapp, tab)
    judged = {row.instance_id for row, _said in tab._judged}
    assert judged, "nothing was judged, so this would pass vacuously"

    # 0.9 is chosen against the fixture, not guessed: this artifact declares
    # `higher_is_better` with scores in 0.63–1.0, and 0.9 leaves 2 of 30 instances. A cutoff
    # that keeps everything would let an artifact-backed list pass this.
    run.set_cutoff("q01", 0.9)
    kept = {r.instance_id for r in run.artifact().instances if r.source_label == "q01"}
    assert len(kept) < len(
        judged
    ), "the cutoff did not exclude anything that was judged"

    tab._pool_cache = None
    # Cleared first, or the drain below is satisfied by the *previous* read and the assertion
    # runs against the list as it was before the cutoff moved.
    tab._judged = []
    tab.reload()
    assert _drain(qapp, lambda: bool(tab._judged)), "the list never came back"
    assert judged <= {row.instance_id for row, _said in tab._judged}


def test_the_scopes_partition_what_was_judged(qapp, tmp_path):
    """An unclear is dropped from the estimate, so its own scope is how you find and settle
    them. Deliberately no yes/no scope: re-watching only your rejections moves precision in
    one direction, which is the figure this tab exists to measure honestly."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    _finish_batch(qapp, tab, answers=[True, False, None])
    judged = tab._scope_rows("judged")
    unclear = tab._scope_rows("unclear")
    assert unclear, "the fixture produced no unclears"
    assert {said for _row, said in unclear} == {None}
    assert len(unclear) < len(judged)
    # Against the drawn batch's ids, not `len(_batch)`: `_scope_rows("batch")` is a
    # comprehension over `_batch`, so the length check could not fail.
    assert {row.instance_id for row, _s in tab._scope_rows("batch")} == {
        row.instance_id for row in tab._batch
    }
    assert {scope for scope, _text in SCOPES} == {"batch", "judged", "unclear"}


def test_an_empty_scope_is_greyed_with_its_count_not_hidden(qapp, tmp_path):
    """The count is the point — it says how much sample an unclear is costing you."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    _finish_batch(qapp, tab, answers=[True, False])
    assert not tab._scopes["unclear"].isEnabled()
    assert not tab._scopes["unclear"].isHidden()
    assert "(0)" in tab._scopes["unclear"].text()


def test_the_list_never_shows_a_score(qapp, tmp_path):
    """The revision rule, on the other surface: these rows are re-judgeable from here, so a
    score beside one you are about to reconsider is exactly the bias `_reveal` withholds.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    _finish_batch(qapp, tab)
    assert tab._judged, "nothing judged, so this would pass vacuously"
    # Pin the *shape*, not one formatting: the list is two columns by construction, so a third
    # carrying a score cannot appear. Adding one used to pass — the check looked for `.3f` and
    # a score written any other way walked past it.
    assert tab._list_model.columnCount() == 2
    assert [
        tab._list_model.horizontalHeaderItem(c).text()
        for c in range(tab._list_model.columnCount())
    ] == ["Clip", "Said"]
    _assert_no_score_visible(tab)


def test_the_list_fills_its_column(qapp, tmp_path):
    """573px of table holding 248px of content left 325px of dead grid on every row.

    The slack goes to the **clip**, which is the column that needs it. It used to stretch the
    last section instead, which is `Said` — one of three words, never wider than "unclear" —
    so the spare width landed on the narrow column while the clip sat at the 100px its header
    cell gave it and elided a recording's name and its timestamp both (measured: that text
    asks for 336px). Either way there is no dead grid, which is what this has always been for.
    """
    from tests.gui.layout_checks import locked_columns

    _exp, _run, tab = _tab(qapp, tmp_path)
    _finish_batch(qapp, tab)
    header = tab._list.horizontalHeader()
    # Not `Stretch`: a stretched clip column could not be dragged, and `Said` grew leftward.
    assert locked_columns(tab._list) == []
    assert header.sectionSize(0) > header.sectionSize(1), (
        f"the clip column is {header.sectionSize(0)}px and `Said` "
        f"{header.sectionSize(1)}px — the slack went to the wrong one"
    )
    assert tab._list.maximumWidth() > 10_000, "the list was capped like a summary table"


def test_continue_offers_exactly_the_clips_left_unjudged(qapp, tmp_path):
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    pending = {row.instance_id for row in tab._queue[1:]}
    tab._decide(True)
    tab.reload()
    assert _drain(qapp, lambda: tab._pending > 0)
    # Against the pending clips, not the code's own f-string rebuilt from the same variable.
    assert tab._pending == len(pending)
    assert tab._continue.text() == f"Continue — {len(pending)} unjudged"

    tab._continue_batch()
    assert {row.instance_id for row in tab._queue} == pending


#: A search short enough that its audit pool is under two batches. The pool is cutoff-free,
#: so only a shorter search makes it smaller.
_SMALL_POOL_ROWS = 300


def test_the_draw_button_says_what_is_left_and_greys_out_when_nothing_is(
    qapp, tmp_path
):
    """It used to stay enabled with the pool exhausted, still promising 25, and clicking it
    wrote a sentence into a label under the video that nobody was looking at."""
    _exp, _run, tab = _tab(qapp, tmp_path, n_rows=_SMALL_POOL_ROWS)
    for _ in range(10):
        if not tab._remaining:
            break
        _finish_batch(qapp, tab)
    assert tab._remaining == 0, "the pool never emptied"
    assert not tab._start.isEnabled()
    assert "Every candidate has been reviewed" in tab._exhausted.text()


def test_the_draw_button_never_promises_more_than_is_left(qapp, tmp_path):
    _exp, _run, tab = _tab(qapp, tmp_path, n_rows=_SMALL_POOL_ROWS)
    _finish_batch(qapp, tab)
    assert (
        tab._remaining < BATCH
    ), "the fixture pool is too big for this to mean anything"
    # The whole string, not `in`: "5 more" is a substring of "Review 25 more", so a
    # containment check passes against the very bug this guards.
    assert tab._start.text() == f"Review {tab._remaining} more"


# --------------------------------------------------------------------------- #
# the curve says how much it rests on
# --------------------------------------------------------------------------- #
# `_estimate` names this failure about its own output: "1.00 from n=3 renders identically to
# 1.00 from n=50 and supports nothing". The tables obey it — `format_precision` writes words
# instead of a number below the floor — and the plot did not: ten bins, ten identical dots,
# and no way to tell which one two clips are holding up.
def _bins(n_by_bin, *, instances=6):
    """Synthetic per-bin rows in the shape `precision_by_score_bin` returns."""
    out = []
    for index, n in enumerate(n_by_bin):
        confirmed = n  # all yes, so precision is 1.0 wherever anything was judged
        out.append(
            {
                "bin": index,
                "score_low": index / len(n_by_bin),
                "score_high": (index + 1) / len(n_by_bin),
                "n_instances": instances,
                "n": n,
                "n_confirmed": confirmed,
                "precision": 1.0 if n else None,
                "interval": audit.wilson_interval(confirmed, n),
                "interval_method": "wilson95" if n else None,
                "below_min_n": bool(n) and n < audit.MIN_N_FOR_ESTIMATE,
            }
        )
    return out


def _plotted(plot) -> list:
    return list(plot.getPlotItem().listDataItems())


def test_a_thin_bin_and_a_thick_one_do_not_draw_the_same(qapp, tmp_path):
    """Two markers, the same split `format_precision` makes in words: a rate resting on 3
    clips is drawn hollow, one resting on 30 solid."""
    from detail.gui import plot_theme

    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._draw_curve(_bins([3, 30, 30, 30]))
    # Compared by colour: a hollow marker's hole is filled with the plot's own surface (so the
    # curve does not show through it), and a solid one with the series colour.
    fills = {
        item.opts.get("symbolBrush").color().name()
        for item in _plotted(tab._plot)
        if item.opts.get("symbol")
    }
    theme = plot_theme.current()
    assert len(fills) == 2, "every bin drew the same marker whatever it rests on"
    assert (
        theme.surface.lower() in fills
    ), "nothing was drawn hollow for the bin under the floor"
    assert (
        theme.identity(0).lower() in fills
    ), "nothing was drawn solid for the bins above it"


def test_the_interval_is_drawn_as_a_band(qapp, tmp_path):
    """The number `_estimate` already publishes beside every rate, shown where the claim is."""
    import pyqtgraph as pg

    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._draw_curve(_bins([2, 30, 30, 30]))
    bands = [
        item
        for item in tab._plot.getPlotItem().items
        if isinstance(item, pg.FillBetweenItem)
    ]
    assert bands, "no interval band"


def test_the_band_is_wider_where_the_sample_is_thinner(qapp, tmp_path):
    """The whole point: a bin backed by two clips must *look* uncertain.

    Read off the **drawn** band. The previous version called `wilson_interval` directly and
    compared widths, which is a duplicate of the storage test and passes against a band drawn
    at a constant width — exactly what its own docstring says it exists to rule out.
    """
    import pyqtgraph as pg

    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._draw_curve(_bins([2, 30, 30, 30]))
    band = next(
        item
        for item in tab._plot.getPlotItem().items
        if isinstance(item, pg.FillBetweenItem)
    )
    lows, highs = (c.getData()[1] for c in band.curves)
    thin, thick = highs[0] - lows[0], highs[-1] - lows[-1]
    assert thin > thick * 3, f"the band is {thin:.3f} at n=2 and {thick:.3f} at n=30"


def test_the_strip_counts_the_judged_clips_behind_each_bin(qapp, tmp_path):
    import pyqtgraph as pg

    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._draw_curve(_bins([0, 4, 18, 9]))
    bars = [
        item
        for item in tab._strip.getPlotItem().items
        if isinstance(item, pg.BarGraphItem)
    ]
    assert bars, "no judged-count strip"
    assert sorted(bars[0].opts["height"]) == [
        4,
        9,
        18,
    ], "the bars are not the judged counts"


def test_a_bin_nobody_judged_is_marked_rather_than_silently_absent(qapp, tmp_path):
    """`precision: None` reaches the curve as a NaN and renders as a gap, which reads as
    "nothing here" when it means "candidates here, and you have looked at none of them".
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._draw_curve(_bins([0, 0, 18, 9]))
    ticks = [
        item
        for item in _plotted(tab._strip)
        if item.opts.get("symbol") and len(item.getData()[0]) == 2
    ]
    assert ticks, "the two unjudged bins left no mark at all"


def test_the_strip_shares_the_curve_s_x_axis(qapp, tmp_path):
    """A bar has to sit under its own point, or the panel invites a misreading."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._draw_curve(_bins([4, 18, 9, 12]))
    assert tab._strip.getViewBox().linkedView(0) is tab._plot.getViewBox()


def test_the_panel_is_absent_until_there_is_something_to_plot(qapp, tmp_path):
    """Both halves, not just the curve: a lone axis under a hidden plot is a leftover."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._draw_curve([])
    # `isHidden`, not `isVisible`: a child of an unshown widget is never "visible", so that
    # assertion holds whatever the panel did.
    assert tab._plot.isHidden()
    assert tab._strip.isHidden(), "an axis left under a hidden curve"
    assert tab._curve_heading.isHidden()


def _bins_over(lo, hi, n_by_bin):
    """Bins covering only part of [0, 1] — what real data gives, since `bin_edges` spans the
    scores that exist, not the range the quantity is defined on."""
    out = _bins(n_by_bin)
    for index, row in enumerate(out):
        row["score_low"] = lo + (hi - lo) * index / len(out)
        row["score_high"] = lo + (hi - lo) * (index + 1) / len(out)
    return out


def test_the_confidence_axis_spans_the_whole_range_not_just_the_data(qapp, tmp_path):
    """Confidence is a [0, 1] quantity. Cropped to the bins that happen to exist, a cutoff at
    0.28 sits hard against the left edge and "where in the range am I" cannot be answered —
    and pyqtgraph relabels a sub-1 axis as "300 (x0.001)", a unit this has no business having.

    Shown, then resized: that is the whole reproduction. The plot's x auto-range is on from
    construction — only `setYRange` is called there — so every relayout re-fits x to whatever
    the bins cover. A plot built in a test and never laid out is never relaid, so a fixture
    that skips the show/resize passes on the ViewBox's 0–1 default and proves nothing.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab.resize(1300, 820)
    tab.show()
    qapp.processEvents()
    tab._draw_curve(_bins_over(0.28, 0.97, [12, 12, 12, 12]))
    qapp.processEvents()
    tab.resize(1100, 700)  # a splitter drag
    qapp.processEvents()
    for name, widget in (("curve", tab._plot), ("strip", tab._strip)):
        low, high = widget.getViewBox().viewRange()[0]
        assert (
            low <= 0.0 and high >= 1.0
        ), f"the {name} axis cropped to its data: {low}–{high}"
    tab.hide()


def test_the_axes_carry_no_invented_unit(qapp, tmp_path):
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._draw_curve(_bins_over(0.28, 0.97, [12, 12, 12, 12]))
    for widget in (tab._plot, tab._strip):
        for edge in ("left", "bottom"):
            assert not widget.getAxis(edge).autoSIPrefix


def test_both_plots_start_their_plotting_area_at_the_same_x(qapp, tmp_path):
    """Or a bar no longer sits under its own point, which is the whole reason they share an x."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    # Shown and laid out: an un-laid-out `AxisItem` reports the default 35.0 whatever
    # `setWidth` was given, so comparing two of them compared 35.0 to 35.0 and held with the
    # call removed. Equality alone is not the property either — two axes both stuck at the
    # default are equal and both wrong — so assert the width they were asked for.
    tab.resize(1300, 820)
    tab.show()
    qapp.processEvents()
    tab._draw_curve(_bins([4, 12, 18, 9]))
    qapp.processEvents()
    assert tab._plot.getAxis("left").width() == audit_tab_module._AXIS_PX
    assert tab._strip.getAxis("left").width() == audit_tab_module._AXIS_PX
    tab.hide()


def test_the_screen_says_precision_is_not_recall(qapp, tmp_path):
    """A number labelled only "precision" gets read as accuracy by the person this tab is for.

    The missing half is a deliberate scope decision, not an oversight: recall would need the
    instances the run *did not* find, which is a different and far larger sample than the clips
    this asks anyone to judge. So the screen says what the number is and what it is not, beside
    the number, rather than leaving "0.9 precision" to be quoted as "90% of the rearing".

    Asserted on the whole rendered text rather than on one widget: the point is that a reader
    sees it, not that a particular label holds it.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    text = _labels(tab)
    assert "recall" in text, "the screen never says what it is not measuring"
    # Not a substring check on "precision" alone — that is satisfied by the headline, which is
    # what was already there. The claim is that the *distinction* is stated.
    assert "does not say how many the run missed" in text


# --------------------------------------------------------------------------- #
# the screen it is judged on
#
# The class `f9350f0a` named on the two Browse tabs, at the third site: **a list and a player
# sharing a splitter pane each get the whole pane height.** The list draws as many rows as the
# window is tall, and the block that holds the clip grows with it — so the three buttons that
# judge the clip end up a screen away from it, or below the fold on a shorter one. Measured
# here before the fix, at 1920x1080: 856px of grid for twelve rows, and a video block 815px
# tall with the judgment row at y=958.
# --------------------------------------------------------------------------- #
def _shown_rows(table) -> float:
    """How many rows of ``table`` are actually on screen, from its laid-out height."""
    chrome = table.horizontalHeader().height() + 2 * table.frameWidth()
    return (table.height() - chrome) / max(
        table.verticalHeader().defaultSectionSize(), 1
    )


def _judging_span(tab) -> int:
    """Top of the clip to the bottom of the row that steps past it, in ``tab``'s coordinates.

    The whole block a reviewer works in. It is what grew with the window before the cap: the
    video takes its height from the width it is handed, so a wider pane pushed the buttons
    down as well as out.
    """
    from PyQt6.QtCore import QPoint

    nav = tab._back.parentWidget()
    top = tab._stack.mapTo(tab, QPoint(0, 0)).y()
    return nav.mapTo(tab, QPoint(0, nav.height())).y() - top


def test_the_clip_list_stops_at_twenty_rows_and_scrolls(qapp, tmp_path):
    """Reported as "the clip table on the left can be much shorter so that it fits on a page".

    A splitter hands each child its whole height whatever it prefers, so the list filled: at
    1920x1080 it drew 28 rows of grid for 12 rows of data, and the rest of the sidebar had
    nowhere to go. Capping its *maximum* is the wrong fix and is recorded as such — a splitter
    is only as tall as its shortest-capped child.
    """
    from detail.gui.tables import BROWSE_ROWS

    _exp, _run, tab = _tab(qapp, tmp_path)
    _finish_batch(qapp, tab)
    tab.resize(1600, 1200)
    tab.show()
    qapp.processEvents()
    try:
        rows = tab._list_model.rowCount()
        assert rows > BROWSE_ROWS, f"only {rows} rows — the cap is not reachable here"
        assert abs(_shown_rows(tab._list) - BROWSE_ROWS) < 0.6, (
            f"the list is {_shown_rows(tab._list):.1f} rows tall in a 1200px window, "
            f"not {BROWSE_ROWS}"
        )
        assert tab._list.verticalScrollBar().maximum() > 0, "the rest must scroll"
    finally:
        tab.hide()


def test_the_clip_and_the_buttons_that_judge_it_do_not_drift_apart(qapp, tmp_path):
    """Reported as "the video and video controls, and the accept, reject buttons should be
    visible and centered and top justified".

    Measured as the one thing that was wrong: the block grew with the window. Before the cap,
    growing the tab by 700px grew it by 700px, because `PictureWithControls` takes the
    picture's height from the width it is handed and the stack was the column's fill child.
    A ratio would not catch it — everything scales together — so this compares the *same*
    span at two window heights.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    tab.resize(1500, 900)
    tab.show()
    qapp.processEvents()
    try:
        short = _judging_span(tab)
        tab.resize(1500, 1600)
        qapp.processEvents()
        assert _judging_span(tab) == short, (
            f"the clip and its buttons spread from {short}px to {_judging_span(tab)}px "
            f"when the window grew by 700"
        )
        assert short < 900, f"the block does not fit a 900px window at {short}px"
    finally:
        tab.hide()


def test_the_clip_player_is_pinned_to_the_top_of_its_block(qapp, tmp_path):
    """`top=True`, the flag `f9350f0a` added to `PictureWithControls` for a player that sits
    beside a list rather than filling a pane of its own.

    Asserted as **wiring**, not as geometry, and deliberately. What the flag does is guarded
    where it lives, parametrised both ways and able to fail
    (`test_image_view.py::test_the_picture_block_centres_unless_told_to_sit_at_the_top`); its
    effect on *this* surface is invisible at the block's usual size, because an aspect-correct
    picture is taller than the block and there is no spare room to centre it in. So a geometry
    check here passes with the flag removed — measured — and the only thing this site can get
    wrong is failing to ask for it. That PR reached two of the three players that sit beside a
    list; this was the third, which is the shape the embedder sweep exists for.
    """
    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    assert tab._player._block._top, "this player centres its clip in a pane it shares"


def test_the_clip_fills_the_width_of_its_pane(qapp, tmp_path):
    """The clip takes the whole pane, and the rows under it are centred on that width.

    A capped, centred column was tried first and rejected on sight: a video that stops short
    of the room it has been given reads as a video that could not fill it. So the block is the
    pane, and "centred" is about the judgment row under it, not about the picture.
    """
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QSplitter

    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    tab.resize(1700, 1000)
    tab.show()
    qapp.processEvents()
    try:
        # Measured against the **splitter's** pane, not the stack's immediate parent: a capped
        # block sits inside a wrapper it fills perfectly, so the obvious comparison passes
        # against the very layout this rules out (it did — verified by reverting).
        pane = tab.findChild(QSplitter).widget(1)
        inset = tab._stack.mapTo(pane, QPoint(0, 0)).x()
        slack = pane.width() - inset - tab._stack.width()
        assert inset <= lines(2) and abs(inset - slack) <= 2, (
            f"the clip block is {tab._stack.width()}px in a {pane.width()}px pane, "
            f"inset {inset} / {slack}"
        )
        # The judgment row is centred on that width — equal slack either side of the three
        # buttons, which is what makes them read as belonging to the clip above them.
        row = tab._buttons[0].parentWidget()
        left = tab._buttons[0].mapTo(row, QPoint(0, 0)).x()
        last = tab._buttons[-1]
        right = row.width() - (last.mapTo(row, QPoint(0, 0)).x() + last.width())
        assert abs(left - right) <= 2, f"judgment row slack {left} and {right}"
    finally:
        tab.hide()


@pytest.mark.parametrize("width", [1100, 1500, 1900])
def test_the_block_reserves_no_more_height_than_its_picture_needs(
    qapp, tmp_path, width
):
    """The reported bug: "the yes, no, unclear buttons drop down on window resize".

    `PictureWithControls` draws the picture at its own aspect ratio inside whatever height it
    is handed, so a block that reserves more than the picture needs leaves the difference as a
    gap under the transport — and `top=True` gathers all of it into one place, right where the
    buttons would otherwise be.

    **Parametrized across widths, and that is the point.** The first version used one narrow
    pane, where the picture happens to fill the block and the gap cannot appear however broken
    the rule is. It passed while the query workspace — the only one wide enough to expose it —
    sat on a 71px gap the whole time, and the user reported it a second time. A guard for a
    threshold has to straddle the threshold.
    """
    from PyQt6.QtCore import QPoint

    _exp, _run, tab = _tab(qapp, tmp_path)
    tab._start_batch()
    tab.resize(width, 1000)
    tab.show()
    qapp.processEvents()
    try:
        transport = tab._player._scrubber
        bottom = transport.mapTo(tab, QPoint(0, transport.height())).y()
        below = tab._where.mapTo(tab, QPoint(0, 0)).y()
        assert below - bottom <= lines(2), (
            f"at {width}px: {below - bottom}px of nothing between the transport and the line "
            f"under it"
        )
    finally:
        tab.hide()


# --------------------------------------------------------------------------- #
# §11: the id is the tool's, the name is the user's
# --------------------------------------------------------------------------- #
def test_every_recording_on_the_screen_is_named_not_numbered(qapp, tmp_path):
    """Three surfaces here showed `recording_id`: the clip rows, the line under the video and
    the by-recording table. The names ride `_compute`, off the UI thread with the artifact.
    """
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    # Every recording, because the batch is drawn at random and any of them can be the clip
    # the `_where` line is currently describing.
    for i, recording in enumerate(exp.recordings):
        recording.set_name(f"Fish tank {chr(ord('A') + i)}")
    named = exp.recordings[0]
    tab = AuditTab(exp, run)
    tab.set_behavior("q01")
    assert _drain(qapp, lambda: bool(tab._headline.text()))
    _finish_batch(qapp, tab)
    assert _drain(qapp, lambda: tab._by_recording_model.rowCount() > 0)

    clips = [
        tab._list_model.item(r, 0).text() for r in range(tab._list_model.rowCount())
    ]
    assert clips and all(text.startswith("Fish tank ") for text in clips), clips
    assert not any(text.startswith(named.id) for text in clips), clips

    by_recording = [
        tab._by_recording_model.item(r, 0).text()
        for r in range(tab._by_recording_model.rowCount())
    ]
    assert by_recording and all(n.startswith("Fish tank ") for n in by_recording)
    assert named.id not in by_recording, by_recording

    tab._show_current()
    assert tab._where.text().startswith("Fish tank "), tab._where.text()


def test_a_deleted_recording_falls_back_to_its_id_rather_than_blank(qapp, tmp_path):
    """A run outlives the recordings it read. A missing name is a gap in the map, not a row
    with nothing in the first column."""
    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.0)
    tab = AuditTab(exp, run)
    tab.set_behavior("q01")
    assert _drain(qapp, lambda: bool(tab._headline.text()))
    tab._names = {}
    assert tab._recording_name("recording_404") == "recording_404"


def test_a_behavior_run_keeps_one_column_because_its_id_is_the_tool_s(qapp, tmp_path):
    """A behavior's `q01` is shown on no other surface, so a column of them would be a column
    of noise."""
    _exp, _run, tab = _tab(qapp, tmp_path)
    model = tab._by_label_model
    headers = [model.horizontalHeaderItem(c).text() for c in range(model.columnCount())]
    assert headers == ["Behavior", "precision"], headers
    assert model.item(0, 0).text() == "rearing", "the id leaked into the name column"


def test_the_first_action_is_at_the_top_of_the_tab_not_its_middle(qapp, tmp_path):
    """Reported: "Start Auditing is still centered in the page and not visible when you first
    enter the audit tab for pose, behavior, and query".

    The intro pane opened with a stretch, which centres it — the intent, for one sentence and
    one button in a large pane. A tab inside a `QTabWidget` is as tall as the *tallest* sibling
    tab, and these workspaces' other tabs are tall, so the only action on the page sat hundreds
    of pixels down.

    `assert_top_justified` cannot catch this and is not the guard for it: a leading stretch
    absorbs the slack exactly as it should, so no label spreads. The measure is where the
    content *starts*.
    """
    from PyQt6.QtCore import QPoint

    _exp, _run, tab = _tab(qapp, tmp_path)
    assert tab._body.currentIndex() == 0, "not on the intro, so this proves nothing"
    tab.resize(1400, 1200)
    tab.show()
    qapp.processEvents()
    try:
        top = tab._intro_headline.mapTo(tab, QPoint(0, 0)).y()
        button = tab._intro_start.mapTo(tab, QPoint(0, 0)).y()
        assert top < lines(8), f"the intro starts {top}px down a {tab.height()}px tab"
        assert button < tab.height() // 2, (
            f"the only action on the page is at {button}px of {tab.height()} — it opens "
            f"below the fold of any window shorter than the tab"
        )
    finally:
        tab.hide()


# --------------------------------------------------------------------------- #
# one estimate, read through the API by every surface
# --------------------------------------------------------------------------- #
# This tab computed its own figures from the storage layer, handing it the artifact's instances.
# When the estimate had to change — weighted by score bin, the user's clips left out — the API
# and the CLI could change and this tab would go on printing the old share beside them, with
# nothing failing: two plausible numbers for one audit. So the tab asks the run, as the CLI does.
_ESTIMATORS = (
    "precision_at_cutoff",
    "precision_by_recording",
    "precision_by_score_bin",
)


def _direct_estimator_calls(source: str) -> list[str]:
    """The storage estimators ``source`` calls itself instead of asking the run."""
    import ast

    return [
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _ESTIMATORS
    ]


def test_no_gui_module_computes_an_audit_figure_itself():
    from pathlib import Path

    import detail.gui

    offenders = {
        path.name: calls
        for path in sorted(Path(detail.gui.__file__).parent.rglob("*.py"))
        if (calls := _direct_estimator_calls(path.read_text(encoding="utf-8-sig")))
    }
    assert not offenders, (
        f"{offenders} compute an audit figure from storage; ask the run "
        "(`audit_precision`, `audit_precision_by_recording`, "
        "`audit_precision_by_score_bin`) so the GUI, CLI and API print one number"
    )


def test_the_estimator_scan_would_catch_a_direct_call():
    planted = (
        "from detail.storage import behavior_audit as audit\n"
        "figure = audit.precision_at_cutoff(run_dir, rows, cutoff=0.5)\n"
    )
    assert _direct_estimator_calls(planted) == ["precision_at_cutoff"]
    assert _direct_estimator_calls("run.audit_precision(label='q01')\n") == []


def test_the_tab_shows_the_figure_the_api_reports(qapp, tmp_path):
    """The behavioural half of the scan: what the tab renders is `audit_precision`'s answer."""
    _exp, run, tab = _tab(qapp, tmp_path)
    for i, row in enumerate(run.audit_sample(label="q01", budget=12)):
        run.label_audit(row.instance_id, i % 3 != 0)
    data = tab._compute()
    assert data["estimate"] == run.audit_precision(label="q01", cutoff=0.0)
    assert data["by_recording"] == run.audit_precision_by_recording(
        label="q01", cutoff=0.0
    )
    shown = [b for b in run.audit_precision_by_score_bin(label="q01") if b["n"]]
    assert [b["bin"] for b in data["bins"]] == [b["bin"] for b in shown]
