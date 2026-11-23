"""BROWSE on a behavior scan: clustering's Browse page, given a scan's words (§20/§21).

A scan's Browse is ``LabelBrowseTab`` — the filters, the ethogram, the list and the player that
never stands blank — built on a query run with ``QUERY_TERMS``. What is checked here is what
changes between the two kinds of run and nothing else: which recordings and embedding runs the
page plays through (asked of the run, ``AuthoredRun.data_sources``), and what the page says.

**The class guarded.** BROWSE read the recordings to decode through off the run's state by the
clustering state's own field name, ``state.source_recording_ids``. A query state names its
recordings ``search_recording_ids``, so on a scan the read raised and the page never loaded.
The answer to "which recordings and embedding runs did this run read" now has one shape for
every authored run kind, built once on ``AuthoredRun``; a kind supplies only its recording list.
``test_every_authored_run_kind_answers_data_sources_in_one_shape`` is driven by the kinds that
exist, so a third kind is covered the day it is added.

The run is real (``tests/api/test_query.py::_searched``, the search run in-process rather than
through the executor), built once for the module and copied per test.
"""

import csv
import shutil

import pytest

from detail.api.experiment import Experiment
from detail.core.progress import format_duration
from detail.gui.cluster_workspace import QUERY_TERMS, TEMPORAL_TERMS
from detail.gui.label_browse import LabelBrowseTab
from detail.gui.qt import QItemSelectionModel, Qt
from detail.storage import behavior_output as bo

from tests.api.test_query import _searched, _two_behaviors
from tests.gui.conftest import drain_busy_calls

#: What every authored run answers ``data_sources()`` with — one shape for both kinds.
SOURCE_KEYS = {"recording_ids", "embed_run_ids", "embed_run_id", "track_id", "method"}


@pytest.fixture(scope="module")
def _scan_template(tmp_path_factory):
    """A completed two-behavior search over s1 and s2, no cutoff set. Built once."""
    root = tmp_path_factory.mktemp("query_browse")
    exp, _run = _searched(root, behaviors=_two_behaviors(["s1", "s2"]))
    return exp.dir


@pytest.fixture
def scan(_scan_template, tmp_path):
    """A per-test copy of the searched run: ``(experiment, run)``."""
    copy = tmp_path / _scan_template.name
    shutil.copytree(_scan_template, copy)
    exp = Experiment(copy)
    return exp, exp.query_run("an_01")


def _cut(run, behavior: str = "q01", keep: int = 6) -> float:
    """Set a cutoff that keeps ``keep`` of the behavior's candidates in s1; return it."""
    scores = sorted(
        (r.score for r in run._audit_pool(behavior) if r.recording_id == "s1"),
        reverse=True,
    )
    cutoff = scores[keep - 1]
    run.set_cutoff(behavior, cutoff)
    return cutoff


def _browse(qapp, exp, run) -> LabelBrowseTab:
    tab = LabelBrowseTab(run, exp, QUERY_TERMS)
    drain_busy_calls(qapp)
    assert tab._loaded, "the page never finished reading the run"
    tab.resize(1440, 847)
    tab.show()
    qapp.processEvents()
    return tab


def _kept(run) -> list:
    return bo.bouts(run.artifact().instances)


# --------------------------------------------------------------------------- #
# the page on a scan
# --------------------------------------------------------------------------- #
def test_a_cut_behavior_fills_the_list_the_ethogram_and_the_filters_and_plays(
    qapp, scan
):
    """Every behavior with a saved cutoff, its kept matches in the list and on the ethogram,
    every searched recording a strip, and the first row on the player — decoded through the
    run's own embedding run, which is what the page could not find before."""
    exp, run = scan
    _cut(run, "q01")  # q02 ("grooming") has no cutoff, so it is absent from the output
    kept = _kept(run)
    assert kept and {r.source_label for r in kept} == {"q01"}
    tab = _browse(qapp, exp, run)
    try:
        assert tab._pages.currentIndex() == 1, "the empty page, on a run with matches"
        assert tab._model.rowCount() == len(kept)
        headers = [
            tab._model.headerData(c, Qt.Orientation.Horizontal)
            for c in range(tab._model.columnCount())
        ]
        assert headers == [
            "behavior",
            "recording",
            "start",
            "duration",
            "confidence",
            "audit",
        ]
        assert {tab._model.item(r, 0).text() for r in range(len(kept))} == {"rearing"}
        assert [
            tab._labels.topLevelItem(i).text(0)
            for i in range(tab._labels.topLevelItemCount())
        ] == ["rearing"], "a behavior with no cutoff is listed"
        assert tab._labels.headerItem().text(1) == "matches"
        assert sorted(
            tab._recordings.item(i).text() for i in range(tab._recordings.count())
        ) == ["s1", "s2"]

        # The ethogram: a strip per searched recording, a band per kept match.
        assert len(tab._band_rows) == 2
        assert sum(len(rows) for rows in tab._band_rows) == len(kept)
        assert (
            tab._ethogram_band._title
            == "Ethogram — kept time, click a match to play it"
        )

        periods = bo.period_by_recording(run.artifact().coverage)
        total = format_duration(sum(bo.instance_ns(r, periods) for r in kept) / 1e9)
        assert tab._summary.text() == (
            f"{len(kept)} of {len(kept)} kept matches · {total} of {total}"
        )

        # The player is never blank: the first row, through this run's embedding run.
        first = tab._rows[0]
        assert tab._playing == first.instance_id
        clip = tab._player._clip
        assert (clip.recording_id, clip.start_t_ns) == (first.recording_id, first.t_ns)
        assert clip.embed_run_id == run.state().embed_run_id != ""

        # INTERPRET's lane is clustering's: nothing on a scan asks for it.
        assert tab._lane is None and tab._ethogram._lane is None
    finally:
        tab.hide()


def test_a_scan_with_no_cutoff_says_to_set_one_on_calibrate(qapp, scan):
    """Before any behavior has a cutoff the scan has no output, and the page says what is true
    for a scan — not clustering's "label a node in INTERPRET", a tab a scan does not have.
    """
    exp, run = scan
    tab = _browse(qapp, exp, run)
    try:
        assert tab._pages.currentIndex() == 0
        assert tab._empty_headline.text() == "No behavior has a cutoff yet"
        assert tab._empty_body.text() == (
            "Set one on Calibrate and the matches it keeps appear here, to filter, watch "
            "and export."
        )
    finally:
        tab.hide()


def test_a_cutoff_that_keeps_nothing_is_not_called_no_cutoff(qapp, scan):
    """A cutoff above every match leaves the output empty with the behavior in it. "No
    behavior has a cutoff yet" is then false; the page names the behavior that kept nothing.
    """
    exp, run = scan
    run.set_cutoff("q01", 1.0)
    assert not _kept(run), "the fixture must keep nothing at 1.0"
    assert run.artifact().manifest["vocabulary"] == ["rearing"]
    tab = _browse(qapp, exp, run)
    try:
        assert tab._pages.currentIndex() == 0
        assert tab._empty_headline.text() == "No kept matches"
        assert tab._empty_body.text() == "rearing has no kept matches."
    finally:
        tab.hide()


def test_a_snapshot_of_a_scan_shows_its_own_matches_and_does_not_export(qapp, scan):
    """A pinned calibration renders as it was pinned. The live run is moved on *before* the
    page reads, so a page reading the live run cannot pass for one reading the snapshot.
    """
    exp, run = scan
    _cut(run, "q01", keep=6)
    pinned = len(_kept(run))
    snapshot_id = run.pin(reason="export")
    _cut(run, "q01", keep=3)
    assert len(_kept(run)) != pinned, "the fixture must diverge from the snapshot"

    view = run.view(snapshot_id)
    tab = _browse(qapp, exp, view)
    try:
        assert tab._model.rowCount() == pinned
        assert tab._playing == tab._rows[0].instance_id
        assert tab._player._clip.embed_run_id == run.state().embed_run_id
        assert not tab._export_all.isEnabled()
        assert "snapshot" in tab._export_all.toolTip()
    finally:
        tab.hide()


def test_export_from_a_scans_browse_writes_the_matches_it_names(
    qapp, scan, tmp_path, monkeypatch
):
    """Export selected and Export all hand the one export dialog this run and the instances
    named — and the export targets read a query run's record as they read a clustering run's.
    """
    from detail.api import exporters

    exp, run = scan
    _cut(run, "q01")
    tab = _browse(qapp, exp, run)
    opened = []
    try:
        model = tab._table.selectionModel()
        for row in (0, 2):
            model.select(
                tab._model.index(row, 0),
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows,
            )
        picked = {tab._row_at(0).instance_id, tab._row_at(2).instance_id}
        assert tab._export_selected.isEnabled()

        monkeypatch.setattr(
            "detail.gui.export.open_export",
            lambda parent, subject, scope, **kw: opened.append((subject, scope)),
        )
        tab._export(selected=True)
        tab._export(selected=False)
    finally:
        tab.hide()

    (subject, selected), (_, everything) = opened
    assert (subject.kind, subject.id) == ("analysis_run", run.id)
    assert set(selected.instance_ids) == picked
    assert set(everything.instance_ids) == {r.instance_id for r in _kept(run)}

    for target in ("instances", "clips", "ethogram"):
        found = exporters.target("analysis_run", target)
        assert exporters.unavailable_reason(found, subject) is None, target
    result = exporters.write_export(
        subject,
        [exporters.target("analysis_run", "instances")],
        tmp_path / "out",
        scope=selected,
    )
    with open(result.dest / "instances.csv", encoding="utf-8-sig") as handle:
        written = {row["instance_id"] for row in csv.DictReader(handle)}
    assert written == picked


# --------------------------------------------------------------------------- #
# the guard: one answer to "what did this run read", for every kind
# --------------------------------------------------------------------------- #
def _authored_kinds() -> set[type]:
    """Every authored-run kind the package ships — the subclasses of ``AuthoredRun`` defined
    under ``detail``, less the snapshot views (a view inherits its kind's answer). Scoped to the
    package, so a class a test plants is not counted as a kind."""
    import detail.api.query  # noqa: F401 — defines QueryRun
    import detail.api.temporal_clustering  # noqa: F401 — defines TemporalClusteringRun
    from detail.api.authored_run import AuthoredRun, SnapshotViewMixin

    found, stack = set(), list(AuthoredRun.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls.__module__.startswith("detail.") and not issubclass(
            cls, SnapshotViewMixin
        ):
            found.add(cls)
    return found


def own_data_sources(classes) -> list[type]:
    """The classes below ``AuthoredRun`` that define ``data_sources`` themselves — a second
    shape for the answer every surface reads in one."""
    from detail.api.authored_run import AuthoredRun

    return sorted(
        (
            cls
            for cls in classes
            if cls is not AuthoredRun
            and any(
                "data_sources" in klass.__dict__
                for klass in cls.__mro__[: cls.__mro__.index(AuthoredRun)]
            )
        ),
        key=lambda cls: cls.__name__,
    )


def test_the_shape_guard_would_catch_a_kind_with_its_own_data_sources():
    """The negative case: a kind that answers in its own shape is found."""
    from detail.api.query import QueryRun

    class Planted(QueryRun):
        def data_sources(self) -> dict:
            return {"search_recording_ids": []}

    assert own_data_sources([Planted, QueryRun]) == [Planted]


def test_every_authored_run_kind_answers_data_sources_in_one_shape(
    qapp, scan, authored_cohort
):
    """``data_sources`` is built once, on ``AuthoredRun``; each kind supplies its recordings and
    nothing else. Then, on a real run of every kind and its snapshot view: the same keys, every
    recording named, and an embedding run for each — what BROWSE plays through."""
    from detail.api.authored_run import AuthoredRun

    kinds = _authored_kinds()
    assert own_data_sources(kinds) == [], "a kind answers data_sources in its own shape"
    for kind in kinds:
        assert (
            kind._recording_ids is not AuthoredRun._recording_ids
        ), f"{kind.__name__} does not say which recordings it read"

    exp, query = scan
    _cut(query, "q01")
    temporal = authored_cohort.run("an_01")
    runs = {type(query): query, type(temporal): temporal}
    assert set(runs) == kinds, (
        "an authored run kind with no real run here — add one so its answer is checked: "
        f"{sorted(k.__name__ for k in kinds - set(runs))}"
    )
    for run in runs.values():
        snapshot_id = run.pin(reason="export")
        for handle in (run, run.view(snapshot_id)):
            sources = handle.data_sources()
            assert set(sources) == SOURCE_KEYS, type(handle).__name__
            assert sources["recording_ids"], type(handle).__name__
            assert set(sources["embed_run_ids"]) == set(sources["recording_ids"])
            assert all(sources["embed_run_ids"].values())


def test_clustering_keeps_its_own_words_on_browse():
    """The scan's words are the scan's: clustering's Browse still says what it said."""
    words = TEMPORAL_TERMS.browse
    assert (words.kept, words.label, words.instance) == ("labelled", "label", "bout")
    assert words.empty == "Nothing labelled yet"
    assert "INTERPRET" in words.empty_hint
    assert words.score == ""
