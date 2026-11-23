"""The two analysis result workspaces answer the same questions the same way.

The class: **a capability added to one result page and not the other.** `ClusterWorkspace` (pose
*and* behavior clustering — they share the class, so they cannot drift) and `_QueryWorkspace`
show the same three things about the same artifact, and were edited at different times by
someone looking at one of them. What shipped:

- three `CollapsibleBand`s on one page and three `QSplitter` panes on the other, so a fold on
  one and a drag on the other were the same gesture with two implementations and two names;
- drag bars on two sections out of six, though `ResizeHandle`'s own docstring said "a band you
  can drag beside a band you cannot is the thing that gets reported as a bug";
- a `BROWSE` you could watch on one page and only read on the other;
- the field-of-view outline, the frame's aspect ratio, a working click and a correctly-oriented
  y axis, each landing on the page being worked on and skipping the other for weeks.

Every one was reported by a person opening the app, because every test was written against the
surface being edited. So this file is data-driven over **both** workspaces: a guard that names
one page is a guard that cannot see this class at all.

The rule the parity rests on, and the one to apply to anything added later:

    **Vertical sections are `CollapsibleBand` + drag handle. Horizontal panes are `GripSplitter`.**

That rule is swept over **every page and dialog the construction corpus builds** by the
requirements grid (`requirements.a_split_puts_panes_side_by_side`, `tests/gui/test_requirements.py`),
not only over the two workspaces here: the query-eval explorer kept a vertical `QSplitter` that a
sweep of these two pages could not see (2026-11-09).

**Re-scoped when clustering went node-centric (§22).** The four stacked sections were built for a
table of clusters at a cut; the clustering page now has a tree, a node card and a labels panel
instead, by decision rather than by drift. What is still held to parity is what both pages still
share: a BROWSE you can watch, the no-vertical-splitter rule, and a sweep that sees both pages.

**And re-converged when the behavior scan's CALIBRATE was rebuilt on INTERPRET's shape** (2026-11):
the map across the top (the tree there, the frequency curve here) and every clip playing in one
montage below it, with no stacked sections on either page. The scan page's Ethograms band went to
BROWSE, as clustering's did; its Locations and Interpretation bands went to MEASURES, so their order
and their fold-and-drag stay guarded on the view that has them.
"""

import pytest

from detail.gui.behavior_query import _ClipPlayer
from detail.gui.collapse import CollapsibleBand
from detail.gui.qt import (
    QSplitter,
    receivers,
)

#: The behavior scan's MEASURES sections, by name and in order. Named rather than "whatever each
#: one happens to be called": the ethogram band used to title itself from `terms.unit`, so the
#: same section was "Cluster ethograms" on one run and "Behavior ethograms" on another.
SECTIONS = ("Locations", "Interpretation")


def _bands(widget) -> dict:
    return {b._title: b for b in widget.findChildren(CollapsibleBand)}


def _cluster(qapp, fitted):
    from tests.gui.test_cluster_workspace import _workspace

    _experiment, _handle, workspace = _workspace(qapp, fitted)
    return workspace, workspace._interpret, workspace._browse


def _query(qapp, tmp_path):
    from tests.gui.test_query_workspace import _workspace

    _exp, _run, workspace = _workspace(qapp, tmp_path)
    return workspace, workspace._calibrate, workspace._browse


def _measures(surfaces):
    workspace, _main, _browse = surfaces["behavior scan"]
    return workspace._measures


@pytest.fixture
def surfaces(qapp, fitted, tmp_path):
    """Both workspaces, built for real. Named rather than parametrised because they need
    different fixtures — and a `skip` on one of them would hide exactly this class."""
    return {
        "clustering": _cluster(qapp, fitted),
        "behavior scan": _query(qapp, tmp_path),
    }


def test_the_scan_pages_measures_keep_their_sections(surfaces):
    """The view that still stacks sections stacks these, by these names."""
    measures = _measures(surfaces)
    assert set(_bands(measures)) == set(SECTIONS), sorted(_bands(measures))


def test_both_pages_play_every_clip_under_their_map(surfaces):
    """The shape both pages now share: no stacked sections on the page where a cut is chosen,
    and the clips playing in one montage. The scan page's CALIBRATE was a player beside the curve
    over a one-row strip of still frames and four folding sections, built apart from INTERPRET
    and reported as dead space, a blank player and a single line of clips."""
    from detail.gui.live_montage import LiveMontage

    for name, (_workspace, main, _browse) in surfaces.items():
        assert not _bands(
            main
        ), f"{name}'s main page stacks sections: {list(_bands(main))}"
        assert main.findChildren(
            LiveMontage
        ), f"{name}'s clips do not play in the montage"


def test_the_clustering_page_is_its_tree_and_montage_not_the_bands(surfaces):
    """The clustering page diverged on purpose (§22): the tree, every clip of the selected
    cluster playing under it — not the scan page's four sections.

    The tree has a drag handle and no collapse header (user decision 2026-11-07: "remove the
    collapsible option for the tree, keep the resizable window"). The page *is* the tree and its
    clips, so folding the tree away left a page of nothing to pick from, and the "▼ Tree" header
    was one more control above the thing it named. The handle is the same `ResizeHandle` every
    stacked band drags with, so the gesture is the one taught everywhere else. The Ethograms band
    moved to BROWSE with the selected cluster's lane, and the scan page's other sections must not
    come back here.
    """
    from detail.gui.live_montage import LiveMontage
    from detail.gui.sizing import ResizeHandle

    _workspace, main, _browse = surfaces["clustering"]
    bands = _bands(main)
    assert not bands, f"the tree must not fold; bands: {list(bands)}"
    handle = main._tree_handle
    assert isinstance(handle, ResizeHandle), "the tree must drag"
    assert receivers(handle, handle.dragged) > 0, "the handle resizes nothing"
    assert main.findChildren(LiveMontage)


def test_the_scan_pages_sections_keep_their_order(surfaces):
    """Not merely the same set. The two pages were built at different times, so Measurements
    landed second on one and last on the other — the same section, a different page. Order is
    part of what "the same three sections" means to somebody who learned one of the pages.

    ``findChildren`` walks in construction order, which for these is layout order.
    """
    measures = _measures(surfaces)
    assert list(_bands(measures)) == list(SECTIONS), list(_bands(measures))


def test_every_section_folds_and_drags(surfaces):
    """Both, on every band. Two of six had a drag bar, and which two was an accident of who
    was editing what."""
    for title, band in _bands(_measures(surfaces)).items():
        assert band._handle is not None, f"{title} cannot be dragged"
        band.set_open(False)
        assert band._handle.isHidden(), f"{title} keeps a bar when folded"
        band.set_open(True)
        assert not band._handle.isHidden()


def test_both_browse_tabs_let_you_watch_a_row(surfaces):
    """One page listed instances you could only read. Same job, same artifact, same surface —
    and a row you can watch is how a systematic error stops looking plausible."""
    for name, (_workspace, _main, browse) in surfaces.items():
        assert browse.findChildren(_ClipPlayer), f"{name}'s BROWSE has no player"
        assert browse._table.selectionModel() is not None
        assert hasattr(
            browse, "_show_row"
        ), f"{name}'s BROWSE plays nothing on selection"


def test_no_page_divides_a_vertical_stack_with_a_splitter(surfaces):
    """The rule, enforced: a splitter here means panes side by side.

    A vertical splitter is the mechanism this pass replaced — it cannot free its floor the way
    a folded band does, and it gave one page drag bars and the other fold toggles for the same
    job.
    """
    for name, (_workspace, main, _browse) in surfaces.items():
        vertical = [
            s
            for s in main.findChildren(QSplitter)
            if s.orientation().name == "Vertical"
        ]
        assert not vertical, f"{name} still stacks panes in a vertical splitter"


def test_the_vertical_splitter_rule_is_swept_over_the_whole_corpus():
    """The rule above reaches every page and dialog: it is a registered requirement on the
    ``splitter`` kind, held in every container, and ``splitter`` is a kind the walk classifies —
    so a vertical splitter anywhere the corpus builds fails the grid."""
    from detail.gui import requirements as reqs
    from tests.gui import subjects

    rule = next(
        r for r in reqs.REQUIREMENTS if r.name == "a_split_puts_panes_side_by_side"
    )
    assert rule.kinds == ("splitter",) and rule.containers is None
    assert "splitter" in {name for name, _types in subjects._KINDS}
    with pytest.raises(AssertionError):
        rule.check(rule.counterexample())


def test_the_sweep_covers_both_pages(surfaces):
    """Guard the guard: a fixture that quietly built one surface would make every test above
    a statement about that one page, which is the failure this file exists for."""
    assert len(surfaces) == 2
    for name, (workspace, main, browse) in surfaces.items():
        assert workspace is not None and main is not None and browse is not None, name
