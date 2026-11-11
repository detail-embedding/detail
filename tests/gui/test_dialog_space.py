"""A dialog keeps its content at the top, like every page does.

**The class.** The top-justify rule was written for *pages* — `CLAUDE.md` and
`docs/gui_design.md` both say "**Pages** top-justify … end its top-level layout with
`addStretch()`" — and the guard that enforces it (`tests/gui/test_overview.py`) names a single
page. **No guard in this suite has ever constructed a `QDialog`.** So dialogs grew without the
rule, and a user resizing a launch form watched a word-wrapped label inflate from 30px to 449px
while its neighbours spread down the window.

Eleven dialogs were missing it. That is the same boundary that let the missing `?` help and the
seven-questions-on-the-front-page problem in: every surface a guard reaches follows the house
rules, and every surface no guard constructs has drifted.

**Discovered, not listed.** The dialogs are found by importing `detail.gui` and walking
`QDialog.__subclasses__()`, so a dialog added later is covered the day it exists rather than the
day someone remembers this file. A dialog that cannot be built from the fixture is skipped by name
in `NEEDS_CONTEXT`, with the reason — the same claim-with-a-reason shape as `UNIT_EXEMPTIONS`.

**The predicate is "something takes the slack", not "the last item is a spacer".** The page guard
requires the spacer to be last; every correctly-stretched dialog in this app puts its status label
and button box below the spacer and would fail that test. What matters is whether *anything*
absorbs extra height — a spacer, a stretch factor, or a widget whose size policy asks for room.
"""

import pytest

from tests.gui.conftest import drain_busy_calls, drain_thread_pool  # noqa: E402

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QLabel  # noqa: E402

import detail.gui.app  # noqa: E402,F401  (the gui package itself)
from detail.api.experiment import Experiment  # noqa: E402
from tests.gui import subjects  # noqa: E402
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers module UIs)
from detail.gui.pages import _has_slack  # noqa: E402

#: Dialogs this fixture cannot build, and why. Each is a claim: a name that no longer resolves to
#: a dialog fails `test_every_exemption_still_names_a_dialog`.
NEEDS_CONTEXT = subjects.DIALOGS_NEEDING_CONTEXT


BUILDABLE = subjects.buildable_dialogs()


def _build(cls, tmp_path):
    """Construct `cls` through `subjects.build_dialog` — the one builder, not a second copy.

    This file used to carry its own dispatcher and its own exemption list, and the two drifted
    exactly as two mechanisms for one question always do: #359 taught `subjects` to build the
    spatial dialogs, and **nine** entries here went on excusing dialogs that were by then
    buildable. Nothing failed, because nothing compared the lists.
    """
    experiment = subjects.an_experiment_with_completed_runs(tmp_path)
    recording = experiment.get_recording("recording_001")
    return subjects.build_dialog(cls, experiment, recording)


def test_no_stale_dialog_exemptions(tmp_path, qapp):
    """An entry naming a dialog the builder *can* build is a claim that has stopped being true.

    The failure this closes: the list rotted nine entries deep because there was no check that
    an excuse still excused anything. `PAGES_NEEDING_CONTEXT` has had this guard; its dialog
    twin did not.
    """
    experiment = subjects.an_experiment_with_completed_runs(tmp_path)
    recording = experiment.get_recording("recording_001")
    stale = []
    for cls in subjects.dialog_classes():
        if cls.__name__ not in NEEDS_CONTEXT:
            continue
        built = subjects.build_dialog(cls, experiment, recording)
        if built is not None:
            built.close()
            stale.append(cls.__name__)
    assert not stale, (
        f"these are listed as needing context and build fine: {stale}. Delete the entries — an "
        f"excuse that excuses nothing is one nobody will question."
    )


def test_there_are_dialogs_to_check():
    """Guard the guard: a walk that stops finding dialogs passes vacuously forever.

    **The floor is on the number actually checked**, not on the number discovered. The first
    version asserted only that the walk found 25 dialogs, which counts every dialog *including*
    the exempt ones — so coverage could fall to a single dialog, one `NEEDS_CONTEXT` entry at a
    time, and this test would go on passing. `BUILDABLE` is the set that gets an assertion run
    against it, so that is the set with a floor under it.

    **And the exemption list is a ceiling, not a floor.** The two numbers move independently: a
    dialog added tomorrow raises the first and should leave the second alone. Writing it as a
    ceiling is what makes "exempt it" the visibly expensive option — the list rotted nine entries
    deep precisely because adding one cost nothing.
    """
    # Both floors came down when the scene-geometry work replaced nine spatial dialogs
    # (SpatialDialog and its three subclasses, the two managers, CalibrationDialog and the
    # read-only preview) with one editor. A ratchet is meant to make a drop deliberate, not
    # impossible: this one is the point of that change, not a fixture losing its reach.
    assert (
        len(subjects.dialog_classes()) >= 35
    ), f"only {len(subjects.dialog_classes())} dialogs found; the walk broke"
    # With `NEEDS_CONTEXT` empty this equals the discovery floor above — which is the claim:
    # every dialog the walk finds is a dialog some rule runs over.
    assert len(BUILDABLE) >= 35, (
        f"only {len(BUILDABLE)} dialog(s) are actually built and checked "
        f"({sorted(c.__name__ for c in BUILDABLE)}); the rest are exempt in NEEDS_CONTEXT. "
        f"Raise the fixture's reach rather than lowering this."
    )
    assert not NEEDS_CONTEXT, (
        f"{len(NEEDS_CONTEXT)} dialogs are exempt: {sorted(NEEDS_CONTEXT)}. The list reached zero "
        f"once every entry turned out to need a dict, a list of rows, or one real artifact — and "
        f"six of the seven were *not compliant*, because a dialog no sweep can build is a dialog "
        f"no rule reaches. Write the fixture in `subjects._providers` rather than adding an entry."
    )


@pytest.mark.parametrize("cls", BUILDABLE, ids=lambda c: c.__name__)
def test_a_dialog_keeps_its_content_at_the_top(qapp, tmp_path, cls):
    dialog = _build(cls, tmp_path)
    layout = dialog.layout()
    assert layout is not None, f"{cls.__name__} has no top-level layout"

    assert _has_slack(layout), (
        f"{cls.__name__} has nothing taking the vertical slack, so resizing it taller spreads "
        f"its rows down the window. Add the button box through `Dialog.set_buttons`."
    )


@pytest.mark.parametrize("cls", BUILDABLE, ids=lambda c: c.__name__)
def test_a_tall_dialog_does_not_spread_its_rows(qapp, tmp_path, cls):
    """Rendered, not structural — the property the user actually sees.

    `_has_slack` is a claim *about* the rendering, and it was wrong in a way the structural
    check could not see: a `QHBoxLayout` whose line edit has stretch 1 counts as slack, so
    `ComparisonExportDialog` got no spacer, passed the structural test, and spread its rows
    down the window anyway. A label never legitimately expands vertically, whatever else on the
    dialog does, so this holds for a filling dialog too.
    """
    dialog = _build(cls, tmp_path)
    dialog.resize(640, 1000)
    dialog.show()
    qapp.processEvents()

    spread = [
        (label.text()[:40], label.height(), label.sizeHint().height())
        for label in dialog.findChildren(QLabel)
        # Word-wrapped labels only, and only past a *multiple* of their hint. A label beside a
        # spin box takes that row's height -- "every frame" renders at 26px against a 15px hint,
        # which is the spin, not spread. The failure this exists to catch is page-scale: the
        # reported one was 30px -> 449px, and `ComparisonExportDialog` was 60px -> 432px. A row's
        # worth of overshoot is layout; an order of magnitude is a dialog with nowhere to put its
        # slack.
        if (
            label.wordWrap()
            and label.isVisible()
            and label.height() > label.sizeHint().height() * 1.5 + 8
        )
    ]

    assert not spread, (
        f"{cls.__name__} stretched {len(spread)} label(s) beyond their natural height in a "
        f"1000px window: {spread}"
    )


def test_a_tall_dialog_does_not_inflate_its_labels(qapp, tmp_path):
    """The number the user actually sees, on the dialog they reported.

    "A spacer exists" is not the property that matters and does not imply it: with
    `addStretch()`'s default factor of 0 the spacer merely has an expanding policy, competes with
    every `Preferred` widget above it, and a word-wrapped `QLabel` grew to 149px anyway — while a
    check for the spacer's presence passed. Assert the height.
    """
    from detail.gui.comparison import ComparisonLaunchDialog
    from tests.api.test_comparison import _cohort
    from tests.gui.test_comparison_gui import TWO_ARMS

    dialog = ComparisonLaunchDialog(_cohort(tmp_path, TWO_ARMS))
    dialog.resize(620, 1000)
    dialog.show()
    qapp.processEvents()

    note = dialog._unit_note
    assert note.height() <= note.sizeHint().height() + 8, (
        f"a two-line note is {note.height()}px tall in a 1000px window "
        f"(its hint is {note.sizeHint().height()}px) — the dialog is spreading"
    )


def test_every_exemption_still_names_a_dialog():
    """A `NEEDS_CONTEXT` entry for a class that no longer exists is a stale claim, and stale
    claims are how the rule went unenforced in the first place."""
    names = {c.__name__ for c in subjects.dialog_classes()}
    stale = sorted(set(NEEDS_CONTEXT) - names)
    assert not stale, f"NEEDS_CONTEXT names dialogs that no longer exist: {stale}"


def test_the_predicate_accepts_a_spacer_that_is_not_last(qapp):
    """The page guard requires the spacer to be the *last* item. Every correctly-stretched dialog
    in this app puts its status label and button box below the spacer, so that predicate would
    fail all of them — which is why this file has its own."""
    from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout

    layout = QVBoxLayout()
    layout.addWidget(QLabel("a row"))
    layout.addStretch()
    layout.addWidget(QPushButton("OK"))

    assert _has_slack(layout)


def test_the_predicate_rejects_a_column_of_small_rows(qapp):
    """The negative case: without it the check above passes on everything."""
    from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout

    layout = QVBoxLayout()
    layout.addWidget(QLabel("a row"))
    layout.addWidget(QPushButton("OK"))

    assert not _has_slack(layout)


def test_a_filling_child_counts_as_slack(qapp):
    """A dialog whose primary content legitimately expands must NOT get a spacer — that is the
    mirror failure, and `add_body_widget`'s docstring records it for pages."""
    from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

    filling = QWidget()
    filling.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    layout = QVBoxLayout()
    layout.addWidget(filling)

    assert _has_slack(layout)


def test_content_that_arrives_after_the_show_is_not_pushed_below_the_spacer(qapp):
    """A dialog that loads off the UI thread is **empty at show**, so the top-justify decision
    is made against nothing.

    The spacer goes in — correctly, for an empty body — and every row the worker's callback adds
    afterwards lands *below* it. Measured on `StreamPreviewDialog`: 150px of dead window above
    the video, and the video's own block sharing what was left with the spacer 50/50, since both
    carried stretch 1. The stretch has to be re-decided once the content it was deciding about
    exists.
    """
    from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

    from detail.gui.dialog import Dialog

    dialog = Dialog()
    layout = dialog.body()
    dialog.show()
    qapp.processEvents()
    assert (
        layout.count() == 1 and layout.itemAt(0).spacerItem() is not None
    ), "an empty body should be top-justified — this test asserts nothing otherwise"

    filling = (
        QWidget()
    )  # what a worker's callback adds: content that takes the slack itself
    filling.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    layout.addWidget(filling, 1)
    layout.addWidget(QLabel("a row under it"))
    qapp.processEvents()  # the re-decide is queued behind the child being parented

    spacers = [
        i for i in range(layout.count()) if layout.itemAt(i).spacerItem() is not None
    ]
    assert not spacers, (
        f"the body still holds a spacer at {spacers} although its content now takes the "
        f"slack; anything added after the show is laid out below it"
    )
    assert layout.itemAt(0).widget() is filling, "the content is not at the top"
    dialog.close()


def test_a_body_filled_after_the_show_with_short_rows_keeps_its_spacer(qapp):
    """The mirror, so the fix above cannot be "never stretch": rows that do *not* take the slack
    still need the spacer, and it still has to be last."""
    from PyQt6.QtWidgets import QLabel

    from detail.gui.dialog import Dialog

    dialog = Dialog()
    layout = dialog.body()
    dialog.show()
    qapp.processEvents()
    layout.addWidget(QLabel("one short row"))
    qapp.processEvents()

    assert layout.count() == 2
    assert layout.itemAt(0).widget() is not None, "the row should be first"
    assert layout.itemAt(1).spacerItem() is not None, "the spacer should be last"
    dialog.close()


# --------------------------------------------------------------------------- #
# the same class, one property over: a rule written for pages that skipped dialogs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", BUILDABLE, ids=lambda c: c.__name__)
def test_a_dialog_lets_you_select_its_text(cls, qapp, tmp_path):
    """Every label in a dialog can be selected and copied.

    **The class, and it is this file's own one step over.** "All user-facing page text is a
    `QLabel` and is selectable" has been the rule since it was written, and `MainWindow._show`
    implements it — for *pages*. A dialog never goes through `_show`, so not one of them could be
    copied from: the same shape as the top-justify rule above, which was also written for pages
    and also skipped every `QDialog` until there was a shared seam to carry it.

    The reason it matters is small and real — someone reporting a bug has to be able to quote the
    screen back — and it is invisible to every behavioural test, because a label nobody can select
    renders exactly like one they can.

    **Shown, not merely built.** `Dialog` applies the policy in `showEvent`, deliberately: a
    dialog's children are commonly built after its constructor and sometimes filled in later
    still, so show is the last moment before anyone can read the text. This test used to check a
    dialog that had never been shown, and passed only because the since-deleted `sizing.finish_dialog` *also*
    applied it at construction — so it was asserting that helper's side effect rather than the
    guarantee the app makes. Dialogs that now take their button box through `Dialog.set_buttons`
    do not call that helper, and the check went red across thirteen of them while every one was
    correct on screen. A widget that was never shown is not in the state being asserted about.
    """
    from PyQt6.QtWidgets import QLabel

    dialog = _build(cls, tmp_path)
    dialog.show()
    qapp.processEvents()
    # ...and wait for anything the dialog loads off the UI thread. `StreamPreviewDialog` builds
    # every one of its labels in a `run_busy` callback, so without this the sweep found **no
    # labels** and took the `pytest.skip` below — passing vacuously over a real bug (#563). A skip
    # is quieter than a green assertion and reads the same in the summary line.
    drain_busy_calls(qapp)
    drain_thread_pool(qapp)
    try:
        labels = [
            label
            for label in dialog.findChildren(QLabel)
            # `.isNull()`, not `not label.pixmap()`: Qt6 returns a *null* QPixmap rather
            # than None, and a null QPixmap is truthy — so the negation filtered out every
            # label and all 17 dialogs skipped. A sweep that skips everything is green.
            if label.text().strip() and label.pixmap().isNull()
        ]
        if not labels:
            pytest.skip(f"{cls.__name__} shows no text labels")
        unselectable = [
            label.text()[:40]
            for label in labels
            if not (
                label.textInteractionFlags()
                & Qt.TextInteractionFlag.TextSelectableByMouse
            )
        ]
        assert not unselectable, (
            f"{cls.__name__} has text nobody can copy: {unselectable}. Add its button box "
            f"through `Dialog.set_buttons`, which carries this the way it carries the spacer."
        )
    finally:
        dialog.close()


def test_the_selectable_sweep_would_catch_a_plain_label(qapp):
    """Guard the guard: a bare QLabel must read as unselectable, or the sweep is vacuous."""
    from PyQt6.QtWidgets import QLabel

    assert not (
        QLabel("plain").textInteractionFlags()
        & Qt.TextInteractionFlag.TextSelectableByMouse
    )


# --- the screen it is designed for ----------------------------------------------------------- #
#: The target, in **logical** pixels — the resolution after OS scaling, which is what a layout has
#: to fit. A 1920x1080 laptop at Windows' recommended 125% reports 1536x864; a MacBook at default
#: scaling reports 1470x833. 1440x810 sits under every machine in play. `docs/gui_design.md` §3.1
#: carries the reasoning and the market share behind it.
TARGET_W, TARGET_H = 1440, 810

#: What a *window* gets of that: a title bar and the menu bar take roughly 60px of height. A
#: dialog opening taller than this puts its own bottom under the Dock — the recorded class, where
#: Qt clamps `resize()` to a floor and the window grows past the display instead of clipping.
USABLE_H = TARGET_H - 60


def _counterexample_functions(package) -> set[str]:
    """Names passed as ``counterexample=`` in ``requirements.py``.

    These functions exist to be **wrong**: each builds the exact shape a rule rejects, so its
    own rule can be watched biting. One of them resizes a host to 400x900 and says so in a
    comment — "deliberately over-tall, the shape this rule exists to catch" — and the sweep
    below read it as a window somebody meant to open that big, so a guard failed on another
    guard's fixture. Excluded by *name*, collected from the decorator rather than hardcoded,
    so a counterexample added tomorrow is covered; excluding the whole file instead would
    stop checking every real window in it.
    """
    import ast

    names: set[str] = set()
    source = (package / "requirements.py").read_text(encoding="utf-8-sig")
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "counterexample" and isinstance(keyword.value, ast.Name):
                names.add(keyword.value.id)
    return names


def _opening_sizes():
    """Every hardcoded `self.resize(w, h)` under `detail/gui`, with where it is.

    Read from the source rather than by building each dialog, because the number under test is
    the one somebody **typed** — a built dialog reports whatever Qt clamped it to on this
    machine's screen, which is a measurement of the test runner rather than of the code.
    """
    import ast
    import pathlib

    import detail.gui.app

    package = pathlib.Path(detail.gui.app.__file__).parent
    skip = _counterexample_functions(package)
    out = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        inside_counterexample = {
            child
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in skip
            for child in ast.walk(node)
        }
        for node in ast.walk(tree):
            if node in inside_counterexample:
                continue
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "resize"
                and len(node.args) == 2
                and all(isinstance(a, ast.Constant) for a in node.args)
                and all(isinstance(a.value, int) for a in node.args)
            ):
                out.append(
                    (
                        f"{path.relative_to(package)}:{node.lineno}",
                        node.args[0].value,
                        node.args[1].value,
                    )
                )
    return out


def test_no_window_opens_bigger_than_the_screen_we_design_for():
    """A window that opens taller than the display puts its own bottom under the Dock.

    Qt clamps `resize()` to a widget's floor and the window *grows past the display* rather than
    clipping — the recorded class from `test_scroll.py`. #362 gave dialogs a scroll, so their
    **floors** are fine now; what nothing checked was the size they **open at**, and three had
    drifted over: 1040x800, 1120x780, 1240x760 against a usable 750.

    The target itself is the thing that had never been decided — it sat open across three
    meetings as 1366x768 versus 1600x900, so "adaptable to a standard screen" was believed rather
    than tested. It is 1440x810 now, and this is what makes that a fact.
    """
    over = [
        (where, w, h)
        for where, w, h in _opening_sizes()
        if w > TARGET_W or h > USABLE_H
    ]
    assert not over, (
        f"{len(over)} window(s) open larger than {TARGET_W}x{TARGET_H} "
        f"(usable height {USABLE_H}): "
        + "; ".join(f"{where} at {w}x{h}" for where, w, h in over)
        + ". Open smaller — the content scrolls, so a generous opening size buys nothing and "
        "costs a window the user has to move before they can reach its bottom."
    )


def test_there_are_opening_sizes_to_check():
    """Guard the guard: a scan that finds nothing passes forever."""
    sizes = _opening_sizes()
    assert len(sizes) >= 15, (
        f"only {len(sizes)} hardcoded opening sizes found — the scan has stopped seeing them, "
        "and the check above is vacuous"
    )


def test_the_counterexample_exclusion_does_not_swallow_real_windows():
    """Guard the exclusion, both ways.

    An exclusion is a hole, and a hole that widens silently is worse than the failure it was
    cut for. Two halves: it must actually name something (or it is dead code pretending to be
    a rule), and the sweep must still see the real windows — excluding the whole
    ``requirements.py`` file, the easy version of this fix, would have stopped checking every
    genuine window in it.
    """
    import pathlib

    import detail.gui.app

    package = pathlib.Path(detail.gui.app.__file__).parent
    skip = _counterexample_functions(package)
    assert skip, "no counterexample= names found — the exclusion excuses nothing"

    sizes = _opening_sizes()
    assert len(sizes) >= 10, f"the sweep stopped finding windows: {sizes}"
    files = {where.split(":")[0] for where, _w, _h in sizes}
    assert (
        len(files) >= 5
    ), f"the sweep collapsed to too few files to be meaningful: {sorted(files)}"


def test_the_over_tall_sweep_would_catch_one():
    """And the sweep itself still bites, on a window that is not a counterexample."""
    over = [(w, h) for w, h in [(TARGET_W + 1, 100), (100, USABLE_H + 1)]]
    assert all(w > TARGET_W or h > USABLE_H for w, h in over)
    # The real check, in the form the test above applies it.
    assert [
        (where, w, h)
        for where, w, h in [("made_up.py:1", 400, 900)]
        if w > TARGET_W or h > USABLE_H
    ] == [("made_up.py:1", 400, 900)]
