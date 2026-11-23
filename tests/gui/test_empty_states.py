"""An index page with nothing in it says so, rather than showing bare column headers.

**The class.** A `QTableView` with zero rows still draws its header strip, and
`pages._index_caption` still appends "double-click a row to open" — so a fresh experiment's
collection pages rendered as a broken-looking table inviting a click on rows that were not there.
Twelve surfaces did this. A user opening the new Comparisons section met it on their first visit
and said it was "not a great look", which is the mild version: the screen is actively telling you
to do something impossible.

**How it hid.** The idiom for the fix already existed in ~30 places — the parenthesised lowercase
label that *replaces* a widget (`MeasuresPanel`; `modules/analysis_cluster_eval.py` too, until
it was removed) — and the CLI has printed `(no analysis runs)` and `(no comparisons)` since it
was written. But every one of those is an embedded *result* view, and no test anywhere asserted
an empty state on a top-level index page. Construction smoke tests build these pages against
empty experiments constantly; they just never looked at what came out.

**The rule.** An index page is `pages.IndexTable` plus `_index_caption(..., rows=n)`. This test
builds every one of them against an experiment with nothing in it and asserts what the user reads,
so a page added later is covered the day it appears in `_PAGES`.
"""

import pytest

from detail.api.experiment import Experiment
from detail.gui.comparison import ComparisonsPage
from detail.gui.active_processes import ActiveProcessesPage
from detail.gui.pages import (
    DatasetsPage,
    ExperimentAnalysisPage,
    ExperimentRecordingsPage,
    ExperimentRunsPage,
    IndexTable,
    ModelManagerPage,
    ModelsPage,
    _index_caption,
)
from detail.gui.qt import QLabel, QPushButton


def _settled_active_processes():
    """The Active-processes page with an empty scan already landed.

    Its scan sweeps the **whole workspace**, which in a test session is shared and which earlier
    tests have filled with runs — so building it and waiting would assert against whatever else
    ran today. Delivering an empty result is the state this file is about, and saying so here is
    what makes the assertion mean the page rather than the session.
    """
    import time

    from detail.gui.qt import QApplication

    page = ActiveProcessesPage(on_open=lambda _r: None)
    # The constructor's own scan must land first: delivered after the empty result, it replaced
    # it with the session's runs — under a loaded full run only, where the scan was slowest.
    deadline = time.monotonic() + 30
    while page._reading and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    page._show([])
    return page


#: Every collection page reached from the rail or the banner, built the way the shell builds it,
#: keyed by the class so the coverage check below can compare against the source without
#: constructing anything. A page whose constructor needs more than `(experiment, on_open)`
#: carries a builder instead.
_PAGES = {
    # Its empty state is the one that matters most here: an empty table on this page claims the
    # *workspace* has nothing running, and the scan is off the UI thread, so it is empty for a
    # moment without meaning it. The caption says "reading" until the scan lands.
    "active_processes": (
        ActiveProcessesPage,
        lambda exp: _settled_active_processes(),
    ),
    "recordings": (
        ExperimentRecordingsPage,
        lambda exp: ExperimentRecordingsPage(exp, on_open=lambda _i: None),
    ),
    "runs": (
        ExperimentRunsPage,
        lambda exp: ExperimentRunsPage(exp, on_open=lambda _i: None),
    ),
    "analysis": (
        ExperimentAnalysisPage,
        lambda exp: ExperimentAnalysisPage(exp, on_open=lambda _i: None),
    ),
    "comparisons": (ComparisonsPage, lambda exp: ComparisonsPage(exp, lambda _i: None)),
    "datasets": (DatasetsPage, lambda exp: DatasetsPage(exp, on_open=lambda _i: None)),
    "models": (
        ModelsPage,
        lambda exp: ModelsPage(
            exp, on_open=lambda _i: None, on_open_distill_run=lambda _i: None
        ),
    ),
    "model_manager": (
        ModelManagerPage,
        lambda exp: ModelManagerPage(
            exp,
            on_open_model=lambda _i: None,
            on_open_experiment_model=lambda _i: None,
            on_distill_launched=lambda _i: None,
        ),
    ),
}

#: Classes that build an `IndexTable` but are not collection pages, and why. A claim, checked by
#: `test_every_index_table_page_is_covered` -- `_PAGES` was hand-written, three pages were never
#: in it, and one of them (`ModelManagerPage`) is a banner mode a new user lands on with nothing
#: installed.
NOT_A_COLLECTION_PAGE = {
    "IndexTable": "the widget itself",
    "_Section": (
        "one titled listing inside ActiveProcessesPage, which is the page this covers — every "
        "one of its sections is walked through it (`findChildren`, so a section added later is "
        "covered the day it is added), and covering the part as well would assert twice about "
        "the same widget and let the page itself be dropped without anything failing"
    ),
    "RecordingDetailPage": "a detail page; its tables are sections of one recording",
    "KeypointsManagerPage": (
        "the skeleton library always lists the four built-in presets, so its table is "
        "never empty; its notice (`(no skeletons yet…)`) covers a library that failed "
        "to read the presets, and `test_keypoints_manager.py` asserts the reading state"
    ),
    "RunDetailPage": "a detail page; its tables are one run's outputs",
    "RunResults": (
        "the list inside RunDetailPage — the same table that exemption is about, in its own "
        "widget now that it carries the results too: one run's recordings, not a collection "
        "you fill. Its empty state is asserted where it means something, on the page "
        "(`test_run_detail.py`)"
    ),
}

#: Collection pages whose notice cannot name a button, and why. The model manager's foundation
#: models are not something a user *creates* -- they are installed from a catalog it lists.
NO_ACTION_FILLS_IT = {
    "model_manager": "its tables list a catalog rather than what you made",
    # Nothing on this page starts a run — they are launched from an experiment, which is the
    # whole reason the page exists. Its notice names *where* runs come from instead of a button
    # it does not have, which is the same promise the rule is making.
    "active_processes": "runs are started from an experiment, not from here",
}


def _build(name, exp):
    """Build the page and let it **settle**.

    A page that reads its rows off the UI thread has zero rows for a moment without meaning
    it, and this file is about the state a page comes to rest in. The transitional one is
    guarded separately, by ``test_a_page_still_reading_does_not_claim_to_be_empty`` — the two
    are different claims and neither test can stand in for the other.
    """
    page = _PAGES[name][1](exp)
    _settle(page)
    return page


def _settle(page) -> None:
    from detail.gui.qt import QApplication

    from tests.gui.conftest import drain_until

    app = QApplication.instance()
    drain_until(app, lambda: not getattr(page, "_reading", False), timeout_ms=5_000)
    app.processEvents()


@pytest.fixture
def empty(tmp_path):
    """An experiment with nothing in it — the state every page is built in on day one."""
    return Experiment.create("empty", tmp_path)


def _tables(page) -> list[IndexTable]:
    return page.findChildren(IndexTable)


@pytest.mark.parametrize("name", sorted(_PAGES))
def test_an_empty_index_page_says_so_instead_of_showing_bare_headers(qapp, empty, name):
    page = _build(name, empty)

    tables = _tables(page)
    assert tables, f"{name} has no IndexTable — it still hand-builds its index"
    for table in tables:
        assert not table.view.isVisibleTo(
            page
        ), f"{name} shows an empty table's column headers"
        assert table._notice.isVisibleTo(page), f"{name} shows nothing at all"
        assert table._notice.text().startswith(
            "("
        ), f"{name}'s empty notice is not in the house idiom: {table._notice.text()!r}"


@pytest.mark.parametrize("name", sorted(_PAGES))
def test_an_empty_index_page_does_not_invite_a_double_click(qapp, empty, name):
    """The caption is the other half, and it is composed before the table exists."""
    page = _build(name, empty)

    captions = [
        label.text()
        for label in page.findChildren(QLabel)
        if "double-click a row to open" in label.text()
    ]

    assert (
        not captions
    ), f"{name} tells the user to double-click a row that is not there"


@pytest.mark.parametrize("name", sorted(_PAGES))
def test_the_notice_names_what_would_fill_it(qapp, empty, name):
    """ "(no comparisons)" is honest but inert. These pages are the ones a new user lands on,
    so the notice carries the action that puts something there — which the embedded result
    views deliberately do not, because nothing the user does *directly* fills those.

    **It names the button, not merely twenty characters.** The first version asserted
    `len(text) > 20`, and `"(no distill runs yet)"` is twenty-one — so the one page that named
    nothing passed, by one character. Length is not the property; a notice that sends a user to
    a button also has to send them to a button that is *there*, which is how the clusterers page
    came to advertise a "New clusterer" beside a button labelled "Fit clusterer".
    """
    if name in NO_ACTION_FILLS_IT:
        pytest.skip(NO_ACTION_FILLS_IT[name])
    page = _build(name, empty)
    labels = [
        button.text().replace(" \u2197", "").strip()
        for button in page.findChildren(QPushButton)
    ]
    stems = {label: label.replace("\u2026", "") for label in labels if label}

    for table in _tables(page):
        notice = table._notice.text()
        named = [label for label, stem in stems.items() if stem in notice]
        assert named, (
            f"{name}'s notice does not name any of its own buttons {labels}, so it does not "
            f"say what would fill it: {notice!r}"
        )
        # And names it the way it is *labelled*. A notice advertising "New comparison" beside a
        # button that says "New comparison" sends the reader looking for something not there —
        # which substring matching alone cannot see, since the shorter one is inside the longer.
        wrong = [
            label
            for label in named
            if "\u2026" not in label and f"{label}\u2026" in notice
        ]
        assert not wrong, (
            f"{name}'s notice writes {wrong} with an ellipsis its button does not have: "
            f"{notice!r}"
        )


# --------------------------------------------------------------------------- #
def test_a_page_still_reading_does_not_claim_to_be_empty(qapp, tmp_path):
    """An empty state is a claim about the **data**, so it may only be made once it is read.

    The converse of this file's rule, and it arrived the moment the recordings table started
    reading its rows off the UI thread: for the length of that read the table has zero rows
    without meaning it, and it was showing the settled notice — telling somebody with sixty
    recordings that they had none, under a button offering to import some. On a local disk the
    window is microseconds wide and nothing looked wrong; on the share this is all for, it is
    the first thing you see.

    Found by rendering the page and reading it. Nothing behavioural could see it: the rows
    arrive, the counts are right, and every other assertion in this file is about the state
    the page comes to rest in.
    """
    experiment = Experiment.create("busy", tmp_path)
    for i in range(3):
        experiment.create_recording(f"recording_{i:03d}")

    page = ExperimentRecordingsPage(experiment, on_open=lambda _sid: None)

    notice = _tables(page)[0]._notice.text()
    assert (
        page._reading
    ), "the page settled before it could be asked about the pending state"
    assert "no recordings yet" not in notice, (
        "the table said the experiment was empty while it was still reading it: "
        f"{notice!r}"
    )
    assert notice.startswith("(") and "reading" in notice  # the house idiom, and honest

    # ...and it does settle onto the real thing, so this is not just a permanent "reading".
    _settle(page)
    assert not page._reading
    assert _tables(page)[0].view.isVisibleTo(page)
    assert page._table.view.model().rowCount() == 3


# --------------------------------------------------------------------------- #
# guard the guard
# --------------------------------------------------------------------------- #
def test_a_populated_index_still_shows_its_table_and_the_click_hint(qapp):
    """The assertions above must be capable of failing — a check that only ever sees the empty
    case would pass just as well if `set_rows` never showed anything at all."""
    table = IndexTable(["a", "b"], empty_text="(nothing)")
    table.set_rows([("1", "2")])

    assert table.view.isVisibleTo(table)
    assert not table._notice.isVisibleTo(table)
    assert "double-click a row to open" in _index_caption("1 thing(s)", rows=1).text()


def test_the_chevron_only_exists_where_a_row_does(qapp):
    """`_table` wrote the "›" cue inside its per-row loop, so an empty navigable table rendered
    an empty extra column and a pointing-hand cursor over nothing."""
    table = IndexTable(["a"], empty_text="(nothing)")

    assert table.view.model().rowCount() == 0
    table.set_rows([("x",)])
    assert table.view.model().index(0, 1).data() == "›"


def test_rows_replace_rather_than_accumulate(qapp):
    """These pages rebuild in place as things are created and deleted — the transition that a
    construction-time fix would have missed, and the one where the empty state matters most.
    """
    table = IndexTable(["a"], empty_text="(nothing)")
    table.set_rows([("x",), ("y",)])
    table.set_rows([("z",)])

    assert table.view.model().rowCount() == 1
    table.set_rows([])
    assert table.view.model().rowCount() == 0
    assert table._notice.isVisibleTo(table)


def test_every_index_table_page_is_covered():
    """Derived from the source, not from memory.

    `_PAGES` is a hand-written list, so it was complete on the day it was written and not
    afterwards: three classes build an `IndexTable` and were never in it, including
    `ModelManagerPage` -- a banner mode a new user reaches on day one with nothing installed,
    which is exactly the state this file exists for.
    """
    import ast
    import pathlib

    import detail.gui

    root = pathlib.Path(detail.gui.__file__).parent
    building = set()
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ClassDef) and "IndexTable(" in (
                ast.get_source_segment(source, node) or ""
            ):
                building.add(node.name)

    assert len(building) >= 10, f"the source walk found only {building}; it broke"
    covered = {cls.__name__ for cls, _builder in _PAGES.values()}
    uncovered = sorted(building - covered - set(NOT_A_COLLECTION_PAGE))

    assert not uncovered, (
        f"these build an IndexTable and no empty-state test constructs them: {uncovered}. "
        f"Add them to _PAGES, or to NOT_A_COLLECTION_PAGE with the reason."
    )


def test_every_not_a_collection_entry_still_names_a_class():
    """A stale exemption is how the rule stops being enforced."""
    import ast
    import pathlib

    import detail.gui

    root = pathlib.Path(detail.gui.__file__).parent
    names = {
        node.name
        for path in root.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig")))
        if isinstance(node, ast.ClassDef)
    }
    stale = sorted(set(NOT_A_COLLECTION_PAGE) - names)
    assert (
        not stale
    ), f"NOT_A_COLLECTION_PAGE names classes that no longer exist: {stale}"


def test_active_processes_shows_no_empty_notice_before_it_has_read(qapp):
    """The unsettled half of the class, which the settled check above cannot see.

    `IndexTable` swaps in its notice at zero rows, and a section starts at zero rows because the
    scan runs off the UI thread — so for as long as the read takes, the page said "Reading every
    experiment…" and "(nothing is running)" one line apart, with the wrong one being the larger.
    Invisible on a local disk (8ms) and the first thing somebody with sixty runs sees on a share.

    `isHidden`, not `isVisible`: nothing here has a shown ancestor, so `isVisible()` is False for
    a table that is perfectly well on screen — the recorded idiom that passes against its own bug.
    """
    from detail.gui.active_processes import ActiveProcessesPage

    page = ActiveProcessesPage(on_open=None)
    assert page._reading, "the fixture raced the scan; this asserts nothing"
    for section in (page._runs, page._exports):
        assert section._table.isHidden(), (
            "an empty-state notice is on screen before the workspace has been read — it is a "
            "claim about the data, and no data has arrived"
        )


def test_active_processes_shows_the_table_once_it_has_read(qapp):
    """The other half. A section that hid its table and never brought it back would pass the
    test above forever while showing nothing at all."""
    from detail.gui.active_processes import ActiveProcessesPage

    page = ActiveProcessesPage(on_open=None)
    page._show([])

    for section in (page._runs, page._exports):
        assert not section._table.isHidden()
