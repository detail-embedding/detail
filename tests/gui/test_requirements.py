"""Every structural requirement, run over every widget in the app.

The list of rules is `detail/gui/requirements.py`; the list of widgets is `subjects.py`. This
file is the grid where they meet, plus the guards that keep either from quietly emptying.

**Why a grid and not another file per rule.** `tests/gui/` had 88 files, 68 guard-shaped, and no
single place saying what a page must satisfy — so a new surface was checked by whichever guards
happened to fail, in whatever order, and a new *kind* of element was checked by nothing because
there was no list to add it to. An inventory (#349) found that of 44 written-down GUI rules only
**16** were actually swept over every subject; 13 more were spot-checked on one or two instances
and read, in the guidelines, exactly like the swept ones.

**The failure mode this file must not have** is the one it exists to catch. Three guards:

- ``test_there_are_subjects_to_check`` floors the number of widgets **built**, never the number
  discovered — `test_dialog_space` learned that one the hard way, since a floor on the discovered
  count lets coverage fall to nothing one exemption at a time.
- ``test_every_requirement_rejects_its_counterexample`` runs each rule against something it must
  fail. A predicate gutted to ``pass`` is caught here rather than by nobody.
- ``test_every_page_the_shell_opens_is_a_root`` derives the roots from `app.py` rather than
  trusting the list, so a page added tomorrow is covered the day it is wired up.
"""

import pathlib

import pytest

pytest.importorskip("PyQt6")

from detail.gui import comparison as gui_comparison  # noqa: E402
from detail.gui import pages as gui_pages  # noqa: E402
from detail.gui import requirements as reqs  # noqa: E402
from detail.storage import locations  # noqa: E402
from tests.gui import subjects as subj  # noqa: E402

PAGE_BUILDERS = subj.page_builders(gui_pages, gui_comparison)
DIALOGS = subj.buildable_dialogs()


@pytest.fixture(scope="module")
def walked(tmp_path_factory, qapp, _temporal_fit_template):
    """Every classifiable widget under every buildable root, built once for the whole module.

    Module-scoped because building ~30 pages and ~30 dialogs is the expensive part and the rules
    only read. A rule that *mutates* a widget would poison its neighbours — none does, and the
    counterexample check is what would surface it.
    """
    tmp_path = tmp_path_factory.mktemp("conformance")
    import pytest as _pytest

    monkeypatch = _pytest.MonkeyPatch()
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    # Undone however the setup ends: a dialog that fails to build used to raise before the
    # `yield`, leave this patch in place, and send every later test on the worker to this
    # fixture's folder as the experiments root (`test_locations` failed three ways on it).
    alive: list = []
    try:
        experiment = subj.an_experiment(tmp_path)
        # Dialogs over a crowded experiment: a pop-up's lists are where rows get hidden, and
        # with one recording nothing is ever long enough to hide any.
        (tmp_path / "crowded").mkdir()
        crowded = subj.a_crowded_experiment(tmp_path / "crowded")
        crowded_recording = crowded.get_recording("recording_001")

        # The built roots are held for the fixture's whole life. A `Subject` holds a *child* widget,
        # and a child does not keep its tree alive: drop the root's Python reference and Qt destroys
        # the C++ side underneath every subject, so the rules meet
        # "wrapped C/C++ object of type QPushButton has been deleted" rather than a button. That is
        # this repo's recorded Qt-lifetime class, met on the first run of this file.
        # Pages are built **the way the shell builds them** — through `MainWindow._show`, which is
        # not decoration: `_show` is where a page's labels are made selectable, so a directly
        # constructed page is not in the state a user meets. Measured: 0 of 263 labels selectable at
        # construction, 105 of 105 after `_show`.
        #
        # One window per page, because `_show` discards the page it replaces; 18 windows cost 0.4s.
        from detail.gui.app import MainWindow

        found, roots = [], []
        for name, build in PAGE_BUILDERS.items():
            window = MainWindow()
            widget = build(experiment)
            window._show(widget, name)
            window.show()  # policies that apply on show — and the state layout rules will need
            alive.extend((window, widget))
            roots.append(name)
            found.extend(subj.walk(widget, name))
        # The shell itself holds one split — the navigation beside the page — which no page
        # walk reaches, because a page is the right-hand pane of it.
        found.append(subj.Subject("splitter", window._split, "MainWindow"))
        roots.append("MainWindow")
        # The clustering page needs a fitted run, which no page builder above has: the
        # session's shared fit, copied, so the walk adds no fit of its own.
        window = MainWindow()
        widget = subj.a_clustering_workspace(
            tmp_path_factory.mktemp("clustering"), _temporal_fit_template, qapp
        )
        window._show(widget, "ClusterWorkspace")
        window.show()
        alive.extend((window, widget))
        roots.append("ClusterWorkspace")
        found.extend(subj.walk(widget, "ClusterWorkspace"))
        for cls in DIALOGS:
            widget = subj.build_dialog(cls, crowded, crowded_recording)
            widget.show()  # a dialog reaches the screen without the shell; show it as one appears
            alive.append(widget)
            roots.append(cls.__name__)
            found.extend(subj.walk(widget, cls.__name__, "dialog"))
        # A dialog's content can land after its show — a list filled, a form swapped — and its
        # re-fit is queued behind that (`Dialog._refit`). Let both happen before anything is
        # measured, as they would before a user could look.
        for _ in range(4):
            qapp.processEvents()
        # The source tree is a subject too. Four house rules are about what the package *never* does,
        # and a rule of that shape cannot be checked by looking at what happens to be built — an icon
        # on a `QListWidgetItem` is not a `QWidget`, so no walk over widgets will ever reach it.
        found.append(subj.PackageSubject())
        roots.append("detail.gui")
        yield found, roots
    finally:
        alive.clear()
        monkeypatch.undo()


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rule", reqs.REQUIREMENTS, ids=lambda r: r.name)
def test_every_subject_meets_the_requirement(walked, rule):
    found, _roots = walked
    failures = []
    for subject in found:
        if not rule.applies_to(subject.kind, subject.container) or not subject.alive:
            continue
        if subject.label in rule.exempt:
            continue
        try:
            rule.check(subject.widget)
        except AssertionError as exc:
            failures.append(f"  {subject.label}: {exc}")

    assert not failures, (
        f"{len(failures)} subject(s) do not meet '{rule.statement}'\n"
        + "\n".join(sorted(failures)[:25])
        + (f"\n  … and {len(failures) - 25} more" if len(failures) > 25 else "")
    )


# --------------------------------------------------------------------------- #
# guard the guard
# --------------------------------------------------------------------------- #
def test_every_requirement_rejects_its_counterexample():
    """A rule that cannot fail passes forever.

    This repo has shipped several: ``len(notice) > 20`` satisfied by a twenty-one character
    string, a parametrisation that iterated zero times, four GUI suites green with a mixin
    gutted. The counterexample is the cheapest possible proof of teeth, so it is a condition of
    registering a rule rather than a good intention.
    """
    toothless = []
    for rule in reqs.REQUIREMENTS:
        try:
            rule.check(rule.counterexample())
        except AssertionError:
            continue
        toothless.append(rule.name)
    assert (
        not toothless
    ), f"these requirements accept their own counterexample, so they cannot fail: {toothless}"


#: A floor on how many requirements are registered — raised as rules migrate, never lowered.
#:
#: Found by *deleting the mechanism* rather than reverting a fix: with ``@requirement`` gutted so
#: nothing registered, **every test in this file passed**. The parametrised grid received zero
#: parameters and was skipped; the counterexample check iterated an empty list and returned. That
#: is exactly the class this file exists to catch — "a parametrisation that iterated zero times"
#: is in CLAUDE.md's recorded list — sitting inside the machinery built to catch it.
MINIMUM_REQUIREMENTS = 15


def test_there_are_requirements_to_run():
    """An empty registry fails loudly rather than passing quietly.

    Without this, deleting the registration decorator is indistinguishable from a green suite: a
    parametrised test over nothing is a *skip*, and a loop over nothing is a *pass*.
    """
    assert len(reqs.REQUIREMENTS) >= MINIMUM_REQUIREMENTS, (
        f"only {len(reqs.REQUIREMENTS)} requirement(s) registered. Either a rule was deleted, or "
        "@requirement stopped registering and every other check in this file is vacuous."
    )


def test_there_are_subjects_to_check(walked):
    """The floor is on what is **built**, not on what is discovered.

    A floor on the discovered count lets coverage fall to nothing one `NEEDS_CONTEXT` entry at a
    time while this file goes on passing — `test_dialog_space`'s recorded lesson, one level up.
    """
    found, roots = walked
    assert len(roots) >= 44, f"only {len(roots)} roots built; the discovery broke"
    by_kind: dict[str, int] = {}
    for subject in found:
        by_kind[subject.kind] = by_kind.get(subject.kind, 0) + 1
    assert by_kind.get("button", 0) >= 140, (
        f"only {by_kind.get('button', 0)} buttons reached across {len(roots)} roots "
        f"({by_kind}); the walk is not seeing the app"
    )


def test_no_requirement_runs_over_nothing(walked):
    """Every rule reaches at least one subject, so none of them passes by never being applied.

    The grid skips a subject whose kind or **container** the rule does not claim, which makes a
    rule silently free the moment either is wrong — ``containers=("pages",)`` for ``"page"`` is a
    typo no reader would catch and no failure would report, and the rule would go on being
    printed into the guidelines as enforced. That is this repo's commonest way for a guard to be
    green and empty: *"a parametrisation over config models iterated zero times for two of
    them"*.
    """
    found, _roots = walked
    empty = {}
    for rule in reqs.REQUIREMENTS:
        n = sum(
            1
            for s in found
            if rule.applies_to(s.kind, s.container)
            and s.alive
            and s.label not in rule.exempt
        )
        if n == 0:
            empty[rule.name] = f"kinds={rule.kinds} containers={rule.containers}"
    assert not empty, (
        "these requirements matched no subject at all, so they cannot fail — check the kind and "
        f"container they claim: {empty}"
    )


def test_every_page_the_shell_opens_is_a_root():
    """Derived from `app.py`, so a page added tomorrow is covered when it is wired up."""
    opens = subj.pages_the_shell_opens()
    assert len(opens) >= 20, f"only {len(opens)} pages found; the AST walk broke"
    uncovered = sorted(opens - set(PAGE_BUILDERS) - set(subj.PAGES_NEEDING_CONTEXT))
    assert not uncovered, (
        f"the shell opens {uncovered} and the conformance walk never builds them. Add a builder, "
        "or an entry in PAGES_NEEDING_CONTEXT with the reason."
    )


def test_no_stale_context_exemptions():
    """A stale claim is how a list stops being complete in the first place."""
    stale = sorted(set(subj.PAGES_NEEDING_CONTEXT) - subj.pages_the_shell_opens())
    assert (
        not stale
    ), f"PAGES_NEEDING_CONTEXT names pages the shell no longer opens: {stale}"


def test_no_stale_rule_exemptions(walked):
    """An exemption that no longer describes a real violation must go, or the list rots."""
    found, _roots = walked
    labels = {s.label for s in found if s.alive}
    stale = {
        f"{rule.name}: {label}"
        for rule in reqs.REQUIREMENTS
        for label in rule.exempt
        if label not in labels
    }
    assert (
        not stale
    ), f"these exemptions name subjects the walk no longer finds: {sorted(stale)}"


def test_the_backlog_only_shrinks():
    """`NOT_YET_SWEPT` is a shrinking list, not a place to put things.

    Every entry is a rule that exists in the guidelines and is *not* enforced over every subject.
    The number is allowed to fall and never to rise; when it reaches zero the guidelines and the
    checks say the same thing, which is the whole point of the exercise.
    """
    assert len(reqs.NOT_YET_SWEPT) <= reqs.BACKLOG_CEILING, (
        f"{len(reqs.NOT_YET_SWEPT)} rules are declared un-swept. Migrating a rule removes its "
        "entry; adding one means a rule was written into the guidelines without a check."
    )
    unreasoned = [k for k, why in reqs.NOT_YET_SWEPT.items() if len(why.strip()) < 10]
    assert not unreasoned, f"these entries carry no real reason: {unreasoned}"


def test_no_swept_rule_is_still_listed_as_un_swept():
    """A rule cannot be both enforced and declared unenforced.

    Migrating a rule is two edits in two places — register the check, remove the backlog entry —
    which makes it exactly the kind of pair that half-happens. The half that gets forgotten is the
    deletion, and it fails silently in the direction that costs the most: the entry goes on reading
    like real remaining work, so the backlog over-reports, the ceiling stops meaning anything, and
    the next person to pick up "the un-swept rules" re-implements a check that already exists.

    It had already happened once here. `a_disabled_button_says_why` was registered and its entry
    stayed behind, saying the rule "waits rather than shipping 63% exempted" while it was being
    enforced over every button in the app.

    So a rule names the entry it retires and the pair is checked, rather than being remembered.
    """
    contradictions = {
        rule.name: rule.retires
        for rule in reqs.REQUIREMENTS
        if rule.retires is not None and rule.retires in reqs.NOT_YET_SWEPT
    }
    assert not contradictions, (
        "these rules are registered and swept, and their NOT_YET_SWEPT entry is still there — "
        f"delete the entry and lower BACKLOG_CEILING: {contradictions}"
    )


def test_the_source_scans_reach_the_real_package():
    """Guard the guard: a scan whose collector finds nothing passes vacuously forever.

    The two package rules added with the dialog sweep both resolve something before they check
    it — one collects the dialog classes, the other the confirmation call sites — and a collector
    that quietly returns an empty set turns its rule into a no-op that reads as a clean pass.
    `_names_used`'s rules cannot fail this way (an empty file set is obvious); these can, because
    the resolution step is where the work is.
    """
    package = pathlib.Path(reqs.__file__).parent
    assert (
        len(reqs._DIALOG_CLASS_NAMES(package)) > 20
    ), "the dialog-class collector found nothing"
    sites = reqs._calls_named(package, "question", on="QMessageBox")
    assert len(sites) > 20, "the confirmation collector found nothing"


def test_dropping_the_declared_exception_fails_the_modal_rule(monkeypatch):
    """And the sharper half: the rule must fail on the real package when its one exception goes.

    A counterexample proves a rule can reject *something* — a synthesized one-file package. This
    proves it rejects something **in this app**, which is the claim that matters and the one a
    counterexample cannot make. Two of this suite's recorded classes are guards that passed
    against their own reverted fix; both would have been caught by asking this question.
    """
    package = pathlib.Path(reqs.__file__).parent
    rule = next(r for r in reqs.REQUIREMENTS if r.name == "a_dialog_opens_modally")
    rule.check(package)  # green with the exception declared

    monkeypatch.setattr(reqs, "_NON_MODAL_DIALOGS", {})
    with pytest.raises(AssertionError, match="analysis_query_eval"):
        rule.check(package)


def test_covered_elsewhere_still_covers():
    """Each `COVERED_ELSEWHERE` entry names a guard that exists, and names it precisely.

    The table exists because the backlog was over-reporting: it listed fifteen rules as
    unchecked when four had a guard somewhere else, so "how much is left" was wrong and the
    ceiling ratcheting it was measuring the wrong number. That fix is only worth anything while
    the pointers stay true — an entry whose test has been renamed or deleted is the same stale
    claim one table over, and this file already carries two other checks of exactly that shape.
    """
    missing = []
    for rule, pointer in reqs.COVERED_ELSEWHERE.items():
        path = pointer.split("::")[0].split(" ")[0]
        if not (pathlib.Path(path).exists()):
            missing.append(f"{rule!r} -> {path} (no such file)")
            continue
        if "::" in pointer:
            name = pointer.split("::")[1].split(" ")[0]
            text = pathlib.Path(path).read_text(encoding="utf-8-sig")
            if f"def {name}" not in text:
                missing.append(f"{rule!r} -> {path} has no {name}")
    assert not missing, (
        "these rules are recorded as covered elsewhere and their guard has moved:\n"
        + "\n".join(missing)
    )


def test_a_rule_is_not_both_covered_and_outstanding():
    """The two tables are a partition. An entry in both reads as done from one side and as
    remaining work from the other, and whichever a reader opens first is the answer they get.
    """
    both = sorted(set(reqs.COVERED_ELSEWHERE) & set(reqs.NOT_YET_SWEPT))
    assert not both, f"listed as both covered and un-swept: {both}"


def test_covered_elsewhere_is_not_a_place_rules_go_to_be_forgotten():
    """Guard the guard: the escape hatch must stay smaller than the thing it relieves.

    A table that means "checked, just not here" is one refactor away from meaning "we decided
    not to check this". Keeping it visibly bounded is what stops the backlog being emptied into
    it — the failure `NOT_YET_SWEPT`'s own ceiling exists to prevent, one level over.
    """
    assert len(reqs.COVERED_ELSEWHERE) <= 6, (
        f"{len(reqs.COVERED_ELSEWHERE)} rules are 'covered elsewhere'. Each is a rule this "
        f"registry does not check; past a handful, the registry stops describing the GUI's "
        f"house style. Write the requirement instead."
    )


# --------------------------------------------------------------------------- #
# The rulebook points here; it does not restate this.
# --------------------------------------------------------------------------- #

#: How many lines CLAUDE.md's GUI Style section may occupy. It is a pointer at this registry and
#: at `docs/gui_design.md`, and a pointer is short. A ratchet, like `BACKLOG_CEILING`: lowered if
#: the pointer gets tighter, raised only as the visible act of putting rules back into the file
#: that cannot check them.
_GUI_SECTION_CEILING = 15


def _gui_style_section(text: str) -> list[str]:
    """The lines of CLAUDE.md's `## GUI Style` section, exclusive of the next heading.

    Pure, so the negative case can hand it a section that grew back.
    """
    lines = text.split("\n")
    start = next(
        (i for i, line in enumerate(lines) if line.startswith("## GUI Style")), None
    )
    if start is None:
        return []
    end = next(
        (
            j
            for j in range(start + 1, len(lines))
            if lines[j].startswith("## ") and not lines[j].startswith("## GUI Style")
        ),
        len(lines),
    )
    return lines[start:end]


def test_the_rulebook_does_not_restate_the_gui_rules():
    """CLAUDE.md's GUI Style section stays a pointer and cannot grow back into a rule list.

    It held **468 lines** restating this registry: a third copy, the one with no test behind it,
    and already the *incomplete* one — six requirements here were never in it, four of them about
    video surfaces. Two rows of CLAUDE.md's own *Recorded classes* are that section's own failures.
    It claimed the window geometry and rail split were persisted while no Qt geometry call existed
    anywhere in `detail/gui/`, and it required sortable headers the whole time that 2 of ~45 tables
    called `setSortingEnabled`. `docs/gui_design.md` had said out loud that three copies existed
    and only one could fail.

    A **line count**, not a content match, because the failure is volume. The section is correct
    while it is a handful of pointers and becomes a drifting copy the moment somebody pastes rules
    back into it; there is no wording to match, only the size of the invitation.
    """
    text = (pathlib.Path(__file__).resolve().parents[2] / "CLAUDE.md").read_text(
        encoding="utf-8-sig"
    )
    section = _gui_style_section(text)
    assert section, (
        "CLAUDE.md has no GUI Style section at all. It should still be there, pointing at this "
        "registry and at docs/gui_design.md — an agent that never learns those exist is the "
        "failure the deletion was not meant to cause."
    )
    assert "requirements.py" in "\n".join(section), (
        "CLAUDE.md's GUI Style section no longer names requirements.py, so nothing sends a reader "
        "to the copy that can fail"
    )
    assert len(section) <= _GUI_SECTION_CEILING, (
        f"CLAUDE.md's GUI Style section is {len(section)} lines, past the "
        f"{_GUI_SECTION_CEILING}-line ceiling. It is a pointer, not a place to restate rules: a "
        "copy there cannot fail, and the one that stood for 468 lines was already missing six "
        "rules this registry enforces."
    )


def test_the_section_ceiling_would_catch_a_pasted_rule_list():
    """Guard the guard: a section grown back into a list of rules must be rejected.

    Without this the check above passes in two worlds — the section is a pointer, or the reader
    stopped finding it — and they are one green test from outside.
    """
    grown = "\n".join(
        ["## GUI Style", "", "See `requirements.py`.", ""]
        + [f"- **Rule {i}.** Prose restating a predicate." for i in range(40)]
        + ["", "## Testing", ""]
    )
    section = _gui_style_section(grown)
    assert len(section) > _GUI_SECTION_CEILING, (
        "a section holding 40 restated rules must exceed the ceiling; the reader is measuring "
        f"something else, and found {len(section)} lines"
    )
    assert (
        _gui_style_section("## Testing\n\nno gui section here\n") == []
    ), "a file with no GUI Style section must read as empty rather than as a passing one"


# --- an exemption reaches only the rule it was written for ------------------------------------
#
# **The class: a file excused from every source scan for one rule's reason.** `_SCAN_EXEMPT` held
# `label_panel.py`, `arena.py` and `__main__.py` because each draws or sets an *icon* — and every
# scan reading that table skipped them for stylesheets, animations, dialogs, playback, splits and
# reloads too. The docstring of `_calls_named` had already recorded one real finding hidden this
# way (a confirmation in `arena.py`); the table kept growing anyway (review, 2026-11-09). Now the
# icon excuses live in `_ICON_EXEMPT`, read by the icon rule alone.


def _package_copy(tmp_path) -> pathlib.Path:
    import shutil

    package = pathlib.Path(reqs.__file__).parent
    copy = tmp_path / "gui"
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))
    return copy


def _plant(package: pathlib.Path, name: str, body: str) -> None:
    path = package / name
    path.write_text(
        path.read_text(encoding="utf-8-sig")
        + f"\n\ndef _planted(widget):\n    {body}\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("name", sorted(reqs._ICON_EXEMPT))
def test_an_icon_excuse_does_not_excuse_a_stylesheet(tmp_path, name):
    """Each icon-excused file is still read by every other scan: a stylesheet planted in it is
    caught, where `_SCAN_EXEMPT` used to skip the file outright."""
    package = _package_copy(tmp_path)
    rules = {r.name: r for r in reqs.REQUIREMENTS}
    rules["the_package_writes_no_stylesheets"].check(package)  # clean before
    rules["the_package_uses_no_icon_sets"].check(package)  # its swatch still excused
    _plant(package, name, 'widget.setStyleSheet("color: red")')
    with pytest.raises(AssertionError, match=name):
        rules["the_package_writes_no_stylesheets"].check(package)


def test_the_scan_wide_exemption_holds_only_the_counterexamples():
    """A reason that belongs to one rule goes in that rule's table; only the file that builds
    every rule's counterexample is skipped by all of them."""
    assert set(reqs._SCAN_EXEMPT) == {"requirements.py"}


# --- each split remembers itself ---------------------------------------------------------------
#
# **The class: a guard that counted a name instead of following the object.** The rule compared
# the number of `GripSplitter(...)` calls in a file with the number of calls *named* `remember`,
# so `clip_popup.py`'s module-level `remember(key, on)` — or a second split's `remember()` — stood
# in for a split that never restored itself (review, 2026-11-09).


def _one_file(tmp_path, body: str) -> pathlib.Path:
    import textwrap

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "page.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return package


@pytest.mark.parametrize(
    "body",
    [
        # a module-level `remember()` beside an unremembered split — what the count accepted
        """
        def remember(key, on):
            pass

        def build():
            split = GripSplitter(Qt.Orientation.Horizontal, key="a")
            remember("toggle", True)
        """,
        # another split's `remember()` covering this one
        """
        def build():
            split = GripSplitter(Qt.Orientation.Horizontal, key="a")
            other = GripSplitter(Qt.Orientation.Horizontal, key="b")
            other.remember()
            other.remember()
        """,
        # the same local name remembered in a different function
        """
        def build():
            split = GripSplitter(Qt.Orientation.Horizontal, key="a")

        def elsewhere(split):
            split.remember()
        """,
        # built and never bound to a name
        """
        def build(row):
            row.addWidget(GripSplitter(Qt.Orientation.Horizontal, key="a"))
        """,
    ],
)
def test_the_split_rule_would_catch_a_split_remembered_by_somebody_else(tmp_path, body):
    package = _one_file(tmp_path, body)
    line = next(
        i for i, text in enumerate(body.splitlines(), start=1) if 'key="a"' in text
    )
    assert reqs._unremembered_splits(package) == [f"page.py:{line}"]


@pytest.mark.parametrize(
    "body",
    [
        """
        def build():
            split = GripSplitter(Qt.Orientation.Horizontal, key="a")
            if not split.remember():
                split.setSizes([1, 2])
        """,
        """
        class Page:
            def build(self):
                self._split = GripSplitter(Qt.Orientation.Horizontal, key="a")

            def restore(self):
                self._split.remember()
        """,
        """
        class Page:
            def build(self):
                split = GripSplitter(Qt.Orientation.Horizontal, key="a")
                self._split = split

            def restore(self):
                self._split.remember()
        """,
    ],
)
def test_the_split_rule_passes_a_split_that_remembers_itself(tmp_path, body):
    assert reqs._unremembered_splits(_one_file(tmp_path, body)) == []
