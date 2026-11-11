"""Every analysis run opens on a landing page, and each result view fills the window (§16).

The class guarded here is **a window spent on chrome instead of on the result.** The run page
used to stack the run's facts, the notes box and a "Results" heading over the workspace, and the
workspace's own tab bar over the view — tree plus clips got 57% of the window. The fix is one
page class for every module: a landing page with a card per view, and a view that fills the
window under one row of tabs, a tab per view, with the way back on the shell's path line
(`<experiment> › <run_id> › <view>`, the run id a link). These tests drive that page through
its own entry point for each shape of module — several views (clustering, behavior scan) and one
(query eval, location export) — so a module cannot quietly fall back to the old stacked page.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QLabel, QPushButton  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui.cluster_workspace import ClusterWorkspace  # noqa: E402
from detail.gui.modules.registry import ANALYSIS_GUI_MODULES  # noqa: E402
from detail.gui.pages import (  # noqa: E402
    VIEW_WAITS_FOR_RUN,
    AnalysisRunDetailPage,
)
from detail.storage import analysis_runs as analysis_store  # noqa: E402

CLUSTER_VIEWS = ["Interpret", "Audit", "Browse", "Measures"]


def _drain(qapp, predicate, timeout_ms: int = 10_000) -> bool:
    from tests.gui.conftest import drain_until

    return drain_until(qapp, predicate, timeout_ms)


def _record(experiment, run_id: str, module_id: str, status: str, **extra) -> None:
    analysis_store.write_analysis_yaml(
        analysis_store.analysis_dir(experiment.dir, run_id),
        {
            "id": run_id,
            "module_id": module_id,
            "recording_ids": ["recording_001"],
            "status": status,
            **extra,
        },
    )


def _location_run(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    _record(
        experiment,
        "an_01",
        "location_export",
        "complete",
        tracking_run_id="track_01",
        artifacts=[
            {"path": "recording_001/tracks.csv", "kind": "tracks", "label": "x"}
        ],
    )
    return experiment, "an_01"


def _landing(page) -> bool:
    return page._stack.currentWidget() is page._landing and page.open_view() is None


# --- one card per view, for every shape of module --------------------------------------------


def test_a_clustering_run_has_a_card_per_view(qapp, fitted_temporal):
    experiment, run_id = fitted_temporal
    page = AnalysisRunDetailPage(experiment, run_id)
    assert _landing(page)
    assert list(page._cards) == CLUSTER_VIEWS
    assert all(card.isEnabled() for card in page._cards.values())
    interpret = page._cards["Interpret"].text().splitlines()
    assert interpret[0] == "Interpret ›"
    assert "behaviors" in interpret[1]  # the temporal module's noun, not "clusters"
    assert interpret[2] == "No labels yet"  # its status line, read from the run


def test_a_behavior_scan_has_a_card_per_view(qapp, tmp_path):
    from tests.api.test_query import _searched

    experiment, run = _searched(tmp_path)
    page = AnalysisRunDetailPage(experiment, run.id)
    assert list(page._cards) == ["Calibrate", "Audit", "Browse", "Measures"]
    assert page._cards["Calibrate"].text().splitlines()[-1] == "1 behavior"


def test_a_location_export_has_one_results_card(qapp, tmp_path):
    experiment, run_id = _location_run(tmp_path)
    page = AnalysisRunDetailPage(experiment, run_id)
    assert list(page._cards) == ["Results"]
    purpose = ANALYSIS_GUI_MODULES["location_export"].purpose
    assert page._cards["Results"].text().splitlines()[1] == purpose
    page._cards["Results"].click()
    assert page.open_view() == "Results"
    # One view: no tab to switch to, so no row of them (the path line names the view).
    assert page._view_tabs is None
    assert page.context_view() == "Results"


def test_a_query_eval_run_has_one_results_card_greyed_while_it_runs(qapp, tmp_path):
    """Greyed rather than absent: the user sees what the run will offer, and why not yet."""
    experiment = Experiment.create("p", tmp_path)
    _record(experiment, "an_01", "query_eval", "running")
    page = AnalysisRunDetailPage(experiment, "an_01")
    assert list(page._cards) == ["Results"]
    card = page._cards["Results"]
    assert not card.isEnabled()
    assert card.toolTip() == VIEW_WAITS_FOR_RUN


def test_a_running_clustering_run_greys_every_planned_card(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    _record(experiment, "an_01", "temporal_clustering", "running")
    page = AnalysisRunDetailPage(experiment, "an_01")
    assert list(page._cards) == CLUSTER_VIEWS
    for card in page._cards.values():
        assert not card.isEnabled() and card.toolTip() == VIEW_WAITS_FOR_RUN
    assert page._result is None and page.open_view() is None


@pytest.mark.parametrize("module_id", ["temporal_clustering", "query"])
def test_the_planned_cards_are_the_views_the_workspace_builds(
    qapp, module_id, fitted_temporal, tmp_path
):
    """The greyed cards come from `planned_views`, the live ones from the built widget: two
    lists of one thing, so this is what keeps them one."""
    if module_id == "query":
        from tests.api.test_query import _searched

        experiment, run = _searched(tmp_path)
        run_id = run.id
    else:
        experiment, run_id = fitted_temporal
    page = AnalysisRunDetailPage(experiment, run_id)
    planned = ANALYSIS_GUI_MODULES[module_id].planned_views()
    assert [view.name for view in page._views] == [name for name, _ in planned]
    assert [view.purpose for view in page._views] == [p for _, p in planned]


# --- opening, switching, going back ----------------------------------------------------------


def _current_tab(page) -> str:
    tabs = page._view_tabs
    return tabs.tabText(tabs.currentIndex())


def _click_tab(page, name: str) -> None:
    """What a click on a tab does: make it current, which the page hears as a switch."""
    tabs = page._view_tabs
    tabs.setCurrentIndex([tabs.tabText(i) for i in range(tabs.count())].index(name))


def test_a_card_opens_its_view_the_row_switches_and_back_returns(qapp, fitted_temporal):
    experiment, run_id = fitted_temporal
    page = AnalysisRunDetailPage(experiment, run_id)
    workspace = page.findChildren(ClusterWorkspace)[0]
    page._cards["Audit"].click()
    assert page.open_view() == "Audit"
    assert page._stack.currentWidget() is page._view_host
    assert workspace.current_view() == "Audit"
    assert _current_tab(page) == "Audit"
    # The workspace's own tab bar is gone: the row is the one switch.
    assert not workspace._tabs.tabBar().isVisibleTo(workspace)
    # And nothing in the row but the tabs: the way back is the shell's path line.
    row = page._view_row.itemAt(0).widget()
    assert row.findChildren(QPushButton) == []
    assert [page._view_tabs.tabText(i) for i in range(4)] == CLUSTER_VIEWS

    _click_tab(page, "Browse")
    assert workspace.current_view() == "Browse" and page.open_view() == "Browse"

    page.close_view()
    assert _landing(page)


def test_the_open_view_survives_a_rebuild(qapp, fitted_temporal):
    """A rebuild — a run finishing, a snapshot opened — must not throw the user back."""
    experiment, run_id = fitted_temporal
    page = AnalysisRunDetailPage(experiment, run_id)
    page._cards["Browse"].click()
    page._build()
    assert page.open_view() == "Browse"
    workspace = page.findChildren(ClusterWorkspace)
    live = [w for w in workspace if w is page._result]
    assert live and live[0].current_view() == "Browse"
    assert _current_tab(page) == "Browse"


def test_a_hand_off_inside_the_workspace_moves_the_row(qapp, fitted_temporal):
    """AUDIT's "Open in INTERPRET" changes the view from inside it; the row must follow."""
    experiment, run_id = fitted_temporal
    page = AnalysisRunDetailPage(experiment, run_id)
    workspace = page._result
    assert _drain(qapp, lambda: workspace._interpret._view is not None)
    page._cards["Audit"].click()
    view = workspace._interpret._view
    workspace._audit.node_requested.emit(int(view.root))
    assert page.open_view() == "Interpret"
    assert _current_tab(page) == "Interpret"
    assert page.context_view() == "Interpret"


def test_a_snapshot_view_says_it_is_read_only_in_the_row(qapp, fitted_temporal):
    experiment, run_id = fitted_temporal
    snapshot_id = experiment.temporal_clustering_run(run_id).pin(reason="export")
    assert snapshot_id
    page = AnalysisRunDetailPage(experiment, run_id)
    page._viewing = snapshot_id
    page._build()
    assert "Measures" not in page._cards  # live numbers never sit under a snapshot
    page._cards["Interpret"].click()
    row_labels = [label.text() for label in page._view_host.findChildren(QLabel)]
    assert f"Read-only · {snapshot_id}" in row_labels


def test_the_landing_page_exports_the_run_and_an_open_view_answers_for_itself(
    qapp, fitted_temporal, monkeypatch
):
    from detail.gui.export import scope_of

    experiment, run_id = fitted_temporal
    page = AnalysisRunDetailPage(experiment, run_id)
    sentinel = object()
    monkeypatch.setattr(page._result, "export_scope", lambda: sentinel)
    assert scope_of(page) is None
    page._cards["Interpret"].click()
    assert scope_of(page) is sentinel


# --- the view fills the window ---------------------------------------------------------------


def _settled(qapp, read, rounds: int = 60) -> bool:
    last = None
    for _ in range(rounds):
        for _ in range(3):
            qapp.processEvents()
        now = read()
        if now == last:
            return True
        last = now
    return False


@pytest.mark.parametrize("size", [(1440, 760), (1920, 900)])
def test_an_open_view_does_not_scroll_the_page(qapp, fitted_temporal, size):
    from detail.gui.scroll import PageScroll

    experiment, run_id = fitted_temporal
    scroll = PageScroll()
    page = AnalysisRunDetailPage(experiment, run_id)
    scroll.setWidget(page)
    scroll.resize(*size)
    scroll.show()
    page._cards["Interpret"].click()
    workspace = page._result
    assert _drain(qapp, lambda: workspace._interpret._view is not None)
    bar = scroll.verticalScrollBar()
    assert _settled(qapp, lambda: (bar.maximum(), page.height()))
    assert bar.maximum() <= 4, f"the page scrolls by {bar.maximum()}px at {size}"
    assert page.height() <= scroll.viewport().height() + 4
    scroll.close()


def test_the_open_view_gets_most_of_the_window(
    qapp, fitted_temporal, monkeypatch, tmp_path
):
    """Through the real shell at 1920x1080: the view in front gets the window, less the
    shell's own chrome and the one view row — no run facts, no notes, no tab bar.

    This measures the *view*, the part this page owns. How much of the view the tree and the
    clips get is INTERPRET's own rows above them (`gui/node_workspace.py`), measured and stated
    in its own tests rather than asserted here against a page that does not lay them out.
    """
    from detail.gui.app import MainWindow
    from detail.storage import locations

    experiment, run_id = fitted_temporal
    # An empty root of its own: the shell scans it, and a shared temp dir holds other tests'
    # deliberately broken experiments.
    root = tmp_path / "experiments_root"
    root.mkdir()
    monkeypatch.setattr(locations, "experiments_root", lambda: root)
    window = MainWindow()
    window.resize(1920, 1080)
    window.show()
    try:
        page = AnalysisRunDetailPage(experiment, run_id)
        window._show(page, run_id)
        page._cards["Interpret"].click()
        workspace = page._result
        interpret = workspace._interpret
        assert _drain(qapp, lambda: interpret._view is not None)
        assert _settled(qapp, lambda: interpret._montage.geometry().bottom())
        top = workspace.mapTo(window, workspace.rect().topLeft()).y()
        share = workspace.height() / window.height()
        assert share >= VIEW_SHARE, (
            f"the view gets {share:.0%} of a 1080px window "
            f"({workspace.height()}px from {top}px)"
        )
    finally:
        window.close()


#: What the open view must get of a 1920x1080 window: everything but the shell's menu, context
#: row and status bar (~100px) and the one view row (~33px).
VIEW_SHARE = 0.85


def test_a_card_is_a_bold_headline_over_plain_lines_and_still_a_button(qapp, tmp_path):
    """Round 2: the view's name reads as a headline (bold, a size up) with its purpose and
    status in normal weight under it, drawn with fonts, not a stylesheet — and the card is
    still one button: a click anywhere on it, or Space with focus, opens the view."""
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest

    experiment, run_id = _location_run(tmp_path)
    page = AnalysisRunDetailPage(experiment, run_id)
    page.resize(1200, 800)
    page.show()
    qapp.processEvents()
    card = page._cards["Results"]
    assert not card.styleSheet() and not page.styleSheet()
    head, purpose = card.headline, card.purpose
    assert head.text() == "Results ›" and head.font().bold()
    assert not purpose.font().bold() and not card.status.font().bold()
    assert head.font().pointSizeF() > purpose.font().pointSizeF()
    assert card.height() >= head.height() + purpose.height(), "the card holds its lines"
    # A click on the purpose line lands on the button, not on the label.
    QTest.mouseClick(
        card, Qt.MouseButton.LeftButton, pos=purpose.geometry().center() + QPoint(0, 0)
    )
    assert page.open_view() == "Results"
    page._show_landing()
    qapp.processEvents()
    card = page._cards["Results"]
    card.setFocus()
    QTest.keyClick(card, Qt.Key.Key_Space)
    assert page.open_view() == "Results"


def test_the_shell_notes_button_shows_the_notes_inside_an_open_view(
    qapp, fitted_temporal, monkeypatch, tmp_path
):
    """With a view open, the shell's Notes button opened the box on the hidden landing page,
    so it did nothing a user could see. The one box moves under the view row while a view is
    open, and back to its place on the landing page when the view closes.

    Open or folded is per place (user, review round 4): opening the notes in a view leaves the
    landing page folded, going back shows it folded unless it was opened there, and each view
    starts folded too."""
    from detail.gui.app import MainWindow
    from detail.gui.notes import NotesBox
    from detail.storage import locations

    experiment, run_id = fitted_temporal
    root = tmp_path / "experiments_root"
    root.mkdir()
    monkeypatch.setattr(locations, "experiments_root", lambda: root)
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    try:
        page = AnalysisRunDetailPage(experiment, run_id)
        window._show(page, run_id)
        qapp.processEvents()
        box = page._notes_box
        assert page.findChildren(NotesBox) == [box], "one box on the page"
        button = window._notes_button
        page._cards["Interpret"].click()
        qapp.processEvents()
        assert button.isVisible() and not button.isChecked()

        button.click()
        qapp.processEvents()
        assert box.isVisible(), "Notes clicked inside a view shows the box"
        assert page._view_host.isAncestorOf(box)
        row_bottom = page._view_row.geometry().bottom()
        assert box.geometry().top() >= row_bottom, "under the view's row"
        assert box.geometry().bottom() <= page._result.geometry().top()

        page._show_landing()  # opened in the view: the landing page is still folded
        qapp.processEvents()
        assert page._landing.isAncestorOf(box) and not box.isVisible()
        assert not button.isChecked(), "the button reads the landing page"
        assert page.findChildren(NotesBox) == [box]

        button.click()  # opened on the landing page too
        qapp.processEvents()
        assert box.isVisible() and button.isChecked()
        page._cards["Browse"].click()  # another view starts folded
        qapp.processEvents()
        assert page._view_host.isAncestorOf(box) and not box.isVisible()
        assert not button.isChecked()
        page._show_view("Interpret")  # the view it was opened in is open again
        qapp.processEvents()
        assert box.isVisible() and button.isChecked()

        page._build()  # and a rebuild with a view open keeps the box and where it is open
        qapp.processEvents()
        assert page._view_host.isAncestorOf(box) and box.isVisible()
        assert page.findChildren(NotesBox) == [box]
        page._show_landing()
        qapp.processEvents()
        assert box.isVisible(), "opened on the landing page before the rebuild"
    finally:
        window.close()


def test_the_path_line_names_the_open_view_and_links_back_to_the_run(
    qapp, fitted_temporal, monkeypatch, tmp_path
):
    """The way back from a view is the run id on the shell's path line, not a button in the
    view row (user, review round 15: `‹ an_32` read as a fifth view). The line follows every
    switch — a tab, a hand-off inside the workspace, the link itself."""
    from PyQt6.QtCore import Qt

    from detail.gui.app import MainWindow
    from detail.storage import locations

    experiment, run_id = fitted_temporal
    root = tmp_path / "experiments_root"
    root.mkdir()
    monkeypatch.setattr(locations, "experiments_root", lambda: root)
    window = MainWindow()
    window.show()
    try:
        page = AnalysisRunDetailPage(experiment, run_id)
        base = f"{experiment.name}  ›  {run_id}"
        window._show(page, base)
        line = window._context
        assert line.text() == base and "<a" not in line.text(), "already there: no link"
        assert (
            line.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
        )

        page._cards["Audit"].click()
        assert f'<a href="landing">{run_id}</a>' in line.text()
        assert line.text().endswith("Audit")
        assert line.text().startswith(experiment.name)

        _click_tab(page, "Browse")
        assert line.text().endswith("Browse")

        workspace = page._result
        assert _drain(qapp, lambda: workspace._interpret._view is not None)
        _click_tab(page, "Audit")
        workspace._audit.node_requested.emit(int(workspace._interpret._view.root))
        assert line.text().endswith("Interpret"), "a hand-off moves the line too"

        line.linkActivated.emit("landing")
        assert _landing(page)
        assert line.text() == base
    finally:
        window.close()
