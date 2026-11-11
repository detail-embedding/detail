"""The live-run panel two pages share, and the band that used to be in it.

**The class this closes.** `RunDetailPage` and `_FitDetailPage` each built the same live-run
scaffold — bar, wrapped status label, `tick`/`finished_run`, a `_start_live` that rebinds the
worker — differing only in which worker they drove. `_FitDetailPage` serves five detail pages, so
a capability added to one copy reached six pages and a capability added to the other reached one,
which is how the two drift. The scaffold is `run_activity.LiveRunPanel` now and both embed it.

**What the band adds that the bar cannot.** The bar says how far along; the status line says what
is happening and is replaced every tick. Neither can say nothing has happened for two hours, and
that is the question someone watching a long run is asking (#341). The band shows the
run's own log, so the last line carries a clock and there is nothing after it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from detail.gui.run_activity import LiveRunPanel  # noqa: E402
from detail.storage import activity  # noqa: E402


def _log(path, rows):
    log = activity.ActivityLog(path, interval_s=0.0)
    for row in rows:
        log.sample(activity.make_row(**row))
    return log


def test_both_live_run_pages_use_the_one_panel():
    """The extraction, asserted rather than trusted.

    Written against the source because the alternative — building both pages and comparing what
    they contain — passes as soon as each *has* a bar, which was true before this change and is
    exactly the state it exists to end.
    """
    import ast
    import pathlib

    import detail.gui.pages as pages_mod

    tree = ast.parse(pathlib.Path(pages_mod.__file__).read_text(encoding="utf-8-sig"))
    builds_a_bar = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and getattr(call.func, "id", "") == "QProgressBar"
            ):
                builds_a_bar.append(node.name)
    assert "RunDetailPage" not in builds_a_bar
    assert "_FitDetailPage" not in builds_a_bar


# --- the band on the five pages that could never fill it ---------------------- #
# The class: **a capability with no way in.** `run_dir` arrived on the shared scaffold with a
# default of `None`, which spells "as before" — so all five `_FitDetailPage` subclasses went on
# not passing it, and every one drew an open Activity band reading "no activity yet" over a run
# that had been writing rows for hours. Nothing failed anywhere: the band built, the page
# rendered, the file was on disk the whole time. The package already had the rule (an input that
# decides whether the output is *correct* has no default) and this is what the default cost.


def test_both_live_run_pages_drive_the_fold_from_their_record():
    """The panel method is only worth having if the pages call it.

    The test above proves `set_active` does the right thing; nothing in it fails if both pages
    stop calling it, which is exactly the state that shipped. Asserted on the source because the
    call sits in `_build`, and reaching `_build` for both pages means two different fixtures for
    a property that is the same one.
    """
    import ast
    import pathlib

    import detail.gui.pages as pages_mod

    tree = ast.parse(pathlib.Path(pages_mod.__file__).read_text(encoding="utf-8-sig"))
    calls_it = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and getattr(call.func, "attr", "") == "set_active"
            ):
                calls_it.add(node.name)

    missing = {"RunDetailPage", "_FitDetailPage"} - calls_it
    assert not missing, (
        f"{sorted(missing)} never folds its band from the record, so a run reached by "
        "navigation shows stop buttons above an empty collapsed band"
    )


def test_the_bar_and_the_status_follow_a_tick(qapp):
    panel = LiveRunPanel()
    panel.set_live(True)

    panel.tick(620, "subject_03_day2 (1/2) — detecting")

    assert panel.progress.value() == 620
    assert panel.status.text() == "subject_03_day2 (1/2) — detecting"


def test_liveness_and_activeness_stay_two_different_facts(qapp):
    """Two facts that were read off one flag, and still must not be merged.

    The bar keys on whether *this page* is streaming the run, because a bar with nothing driving
    it sits frozen at zero — worse than no bar. Whether the run is **going** is a different fact,
    lives in the record, and is what decides whether the page draws the run's parts at all.
    """
    panel = LiveRunPanel()
    panel.set_live(False)  # nobody is streaming this page
    panel.set_active(True)  # ...but the record says it is running

    assert panel._active is True, "the record's verdict was dropped"
    assert not panel.progress.isVisible(), (
        "the bar came back with no worker to move it — a bar frozen at zero reads worse "
        "than no bar, which is why these are two calls and not one"
    )


def test_the_status_line_is_one_line_whatever_it_says(qapp):
    """The vertical half of the jitter.

    It used to wrap, so a status growing from `~3h59m left` to `detecting · 47% · ~19m00s left ·
    ~4h01m run` took a second line and shoved everything below it down — twice a second, for the
    life of the run. Measured 63px for one line of text before this.
    """
    from detail.gui.sizing import lines as em

    panel = LiveRunPanel()
    panel.resize(500, 200)
    panel.show()
    qapp.processEvents()

    heights = set()
    for text in (
        "~3h59m left",
        "subject_03_day2 (1/2) · detecting · 47% · ~19m00s left · ~4h01m run",
        "loading model",
    ):
        panel.tick(100, text)
        panel.layout().activate()
        qapp.processEvents()
        heights.add(panel.status.height())

    assert heights == {em(1)}, f"the status line changed height: {heights}"
