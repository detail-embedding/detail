"""The workspace a clustering run is interpreted in (§21/§22): INTERPRET, AUDIT, BROWSE.

INTERPRET is ``gui/node_workspace.NodeInterpretTab``: the label field and the filters, the merge
tree shaded by what matches them, and every clip of the selected cluster playing under it. AUDIT
is the shared §19 reliability audit and BROWSE the instances the current labels yield; both read
only the artifact, so they follow every label the moment the output catches up.

**Generic over the run, not over the method.** §22 keeps the pose and temporal *method* protocols
apart on purpose, but the interaction is the same job whether a cluster is a posture or a
behavior, so one workspace serves both. ``WorkspaceTerms`` carries the words, because a posture
and a behavior are not the same noun and a screen that called both "cluster" would be the
shared-interface trap in the wording rather than the types.

A **snapshot** renders through the same workspace: its handle satisfies every read and refuses
every write (``editable`` is false), so the controls grey out and say why.
"""

from dataclasses import dataclass

from detail.core.progress import format_duration
from detail.gui.label_check import LabelCheckTab
from detail.gui.label_browse import LabelBrowseTab
from detail.gui.qt import QSize, QSizePolicy, QTabWidget, QVBoxLayout
from detail.gui.result_views import MultiViewWorkspace
from detail.gui.node_workspace import NodeInterpretTab
from detail.gui.tables import top_of_pane


@dataclass(frozen=True)
class BrowseWords:
    """BROWSE's words: what the run's output is, and what to say before there is any.

    BROWSE serves a clustering run and a behavior scan alike, and the two outputs are not the
    same thing: a clustering run's is the time somebody *labelled*, a scan's the matches each
    behavior's cutoff *kept*. Before there is any, the page has to say what is true for the run
    in front of it — labelling a node in INTERPRET means nothing on a scan — so where the output
    comes from is a word here, never a branch on the run's type.
    """

    kept: str = "labelled"  # "12 of 40 labelled bouts", "labelled time"
    label: str = "label"  # one row of the filter
    labels: str = "labels"
    instance: str = "bout"  # one band on the ethogram
    instances: str = "bouts"  # the filter's count column
    empty: str = "Nothing labelled yet"
    empty_hint: str = (
        "Label a node in INTERPRET and its instances appear here, to filter, watch and "
        "export."
    )
    #: The list's score column heading, or "" for none: a scan's matches are chosen by their
    #: confidence, so its list shows and sorts by it; clustering's list has never had one.
    score: str = ""


@dataclass(frozen=True)
class WorkspaceTerms:
    """What this module's clusters are called, on screen.

    §22: the pose module's vocabulary is something you *look at*, the temporal module's is
    something that *unfolds*. Same interaction, different nouns.
    """

    unit: str = "cluster"  # "cluster" / "behavior"
    units: str = "clusters"
    subject: str = "posture"  # what one of them *is*
    # What the size column counts. A posture cluster holds frames; a behavior holds bouts, and
    # calling those "rows" invites reading a 12-bout behavior as twelve frames of behavior.
    member: str = "rows"
    # What the *share* column is a share of — and the reason this is a term at all. It read
    # "% time" for both modules while carrying `share`, which is a fraction of `member`: right
    # enough for postures, where a row is a frame, and simply wrong for behaviors, where a
    # cluster of eight-second bouts and one of half-second bouts can hold an equal share of the
    # vocabulary. The count column was given a term for this exact reason; the one beside it
    # was not. Time now has its own column, from the artifact.
    share: str = "% frames"
    # What the coverage line calls time inside an instance. A posture tiles its recording apart
    # from flickers; bouts leave the time between them unlabelled on purpose, and a line reading
    # "in a behavior" against that would read as a shortfall.
    inside: str = "in a posture"
    browse: BrowseWords = BrowseWords()


POSE_TERMS = WorkspaceTerms(
    unit="cluster", units="clusters", subject="posture", member="rows", share="% frames"
)


TEMPORAL_TERMS = WorkspaceTerms(
    unit="behavior",
    units="behaviors",
    subject="behavior",
    member="bouts",
    share="% bouts",
    inside="in a bout",
)


#: A behavior scan's (§20) words, for the BROWSE it shares with clustering. An instance is a
#: match the search found and a behavior's cutoff kept — not a bout of a cluster somebody named.
QUERY_TERMS = WorkspaceTerms(
    unit="behavior",
    units="behaviors",
    subject="behavior",
    member="matches",
    share="% matches",
    inside="in a match",
    browse=BrowseWords(
        kept="kept",
        label="behavior",
        labels="behaviors",
        instance="match",
        instances="matches",
        empty="No behavior has a cutoff yet",
        empty_hint=(
            "Set one on Calibrate and the matches it keeps appear here, to filter, watch "
            "and export."
        ),
        score="confidence",
    ),
)


#: The workspace's views, in the order the run page's row shows them.
INTERPRET, AUDIT, BROWSE, MEASURES = "Interpret", "Audit", "Browse", "Measures"


def cluster_view_specs(
    terms: WorkspaceTerms, *, editable: bool = True
) -> list[tuple[str, str]]:
    """``(name, purpose)`` per view, without building the workspace.

    The run page's cards need these while the run is still computing (`ModuleUI.planned_views`),
    and the built workspace reads the same list, so the cards and the views cannot disagree.
    Measures is a live-run view only: it reads the live artifact.
    """
    specs = [
        (
            INTERPRET,
            f"Label {terms.units}: the tree, the filters, and every clip playing.",
        ),
        (
            AUDIT,
            "Clean each label: watch its clips, remove the wrong ones, estimate its "
            "precision.",
        ),
        (BROWSE, "Every instance the labels give, and the ethogram per recording."),
    ]
    if editable:
        specs.append((MEASURES, "Time, bouts and transitions per label."))
    return specs


def _count(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def _labels_status(run) -> str:
    """How many labels the run has — one small state read."""
    labels = getattr(run.state(), "labels", None) or {}
    return _count(len(labels), "label", "labels") if labels else "No labels yet"


def _recordings_status(run) -> str:
    recordings = run.data_sources()["recording_ids"]
    return _count(len(recordings), "recording", "recordings")


# --------------------------------------------------------------------------- #
# §22 INTERPRET / BROWSE — where a vocabulary becomes something a person can name
# --------------------------------------------------------------------------- #


def _total_time(cluster: dict) -> str:
    """How much time a cluster accounts for, with the share of observed time beside it.

    A *length*, not a position, so it reads through ``core.progress.format_duration`` (the
    package's one duration formatter) rather than the fixed ``h:mm:ss`` above — "12m07s" beats
    "0:12:07" in a column, and a second hand-rolled formatter for the same quantity is the bug
    that rule exists to stop.

    The share is dropped at zero, and when the artifact observed nothing at all (``time_share``
    is ``None``). A cluster can legitimately account for no time: pose clustering emitted an
    instance only for a run that clears ``min_run_s``, so a vocabulary fine enough that no two
    adjacent frames share a label contributes nothing to the artifact — which is worth seeing,
    since it is also what BROWSE and the audit will be empty of. "0s (0.0%)" says that twice.
    """
    raw = cluster.get("total_ns", 0)
    if raw is None:
        # Not zero — *unknown*. The artifact was projected under a previous id space, so the
        # lookup misses every row; printing `0s` under a real row count reads as a claim about
        # the data. It refreshes on the next granularity change, which is what reprojects it.
        return "—"
    total_ns = int(raw)
    text = format_duration(total_ns / 1e9)
    share = cluster.get("time_share")
    return text if share is None or total_ns == 0 else f"{text}  ({100 * share:.1f}%)"


class _CurrentTabWidget(QTabWidget):
    """A tab widget as tall as the tab in front, not the tallest tab.

    ``QTabWidget`` takes the largest hint of all its pages, and the page scroll gives the
    workspace that height whichever tab shows: AUDIT sat in a page 1063px tall on a 760px laptop
    window because INTERPRET's card wanted 1010, and the clip and its answers ran past the
    bottom. (``QStackedLayout`` would skip a page with an ``Ignored`` policy; ``QTabWidget`` asks
    each page directly and does not.) Re-asked on every tab change, so the page scroll re-fits.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.currentChanged.connect(lambda _index: self.updateGeometry())

    def _current_only(self, full: QSize, hint) -> QSize:
        current = self.currentWidget()
        if current is None:
            return full
        tallest = max(hint(self.widget(i)).height() for i in range(self.count()))
        return QSize(full.width(), full.height() - tallest + hint(current).height())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt virtual
        return self._current_only(super().sizeHint(), lambda w: w.sizeHint())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt virtual
        return self._current_only(
            super().minimumSizeHint(), lambda w: w.minimumSizeHint()
        )

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt virtual
        # A word-wrapped label in any tab turns on height-for-width, and in that path
        # QStackedLayout asks every page again — INTERPRET's 1010px included — so the run page
        # scrolled by 509px with AUDIT in front. The hints above already answer for the tab in
        # front; this keeps Qt using them.
        return False


class ClusterWorkspace(MultiViewWorkspace):
    """§21's authored-run workspace for a clustering vocabulary: INTERPRET, AUDIT, BROWSE.

    Each is a full-window **view** the run page switches between (`MultiViewWorkspace`); the
    workspace keeps them in a tab widget whose bar is hidden. Three views, not §22's four. DIFFERENCES computes through §24's comparison path, which has no
    GUI at all, and shipping it empty would claim more than is here.

    AUDIT is ``gui/label_check.py``: a list of labels and their clusters beside a review pane
    (``gui/label_clean.py``) showing every clip of one cluster, to mark the wrong ones and finish;
    the finish gives the cluster's exact precision. A saved pose-clustering run (the module is
    removed) keeps no clip edits and gets the list with its edits disabled. It gives figures, never
    verdicts. A query run reviews on the same pane in ``gui/query_review.py``, one recording of the
    behavior its selector chose where this has a cluster. MEASURES is the §23 table for this run's artifact, on live runs.
    """

    def __init__(self, experiment, run, terms: WorkspaceTerms = POSE_TERMS) -> None:
        super().__init__()
        self._terms = terms
        # The run-detail page top-justifies and gives each child its size hint, so without this
        # the workspace collapses to its minimum and leaves the lower half of the window empty.
        # `pages.add_body_widget` reads this policy and hands an expanding child the stretch.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # **No `setMinimumHeight` here.** There used to be a 640 against contents that need 933,
        # and one number cannot be a floor and a promise at once: it let Qt squeeze the workspace
        # 293px below what its children need, and they were clipped inside it rather than the
        # page saying it did not fit. That is how the 3-D scatter rendered at 172px against its
        # own declared 420px minimum. Left unset, the floor is the contents' own.
        #
        # **A view open full window does not scroll**, though: the run page asks for no height
        # then (`AnalysisRunDetailPage.minimumSizeHint`, review round 1), so the view is held
        # to the window and anything below its floor is clipped, not scrolled. This comment
        # said the page scrolls, and INTERPRET's floor was 764px in a 660px column at 1440x847
        # — the clip grid's bottom 94px went under the window's edge with no way to reach it.
        # So each view's own floor has to fit a laptop window; INTERPRET's column gives way in
        # `NodeInterpretTab._fit_tree`, and `tests/gui/test_laptop_layout.py` holds every view
        # to it at 1440x847.
        layout = QVBoxLayout(self)
        # Edge to edge: the run page's view host already keeps the margin from the window.
        layout.setContentsMargins(0, 0, 0, 0)
        self._tabs = _CurrentTabWidget()
        self._interpret = NodeInterpretTab(run, experiment, terms)
        self._browse = LabelBrowseTab(run, experiment, terms)
        # The browse list is a view of the artifact, which every authored edit reprojects, so it
        # has to follow rather than hold whatever it read when the tab was built.
        self._audit = LabelCheckTab(experiment, run)
        # Both views read the artifact, which every authored edit reprojects, so both follow
        # rather than hold whatever they read when the tab was built. Renaming a cluster changes
        # what the audit is asking about.
        self._interpret.changed.connect(self._browse.reload)
        self._interpret.changed.connect(self._audit.reload)
        # A label edit here moves clusters in or out of what was audited: the ring and the
        # card must follow without a page reload.
        self._interpret.changed.connect(self._interpret.refresh_audit)
        self._interpret.status_changed.connect(self._audit.reload)
        # The cluster picked on INTERPRET is a lane under BROWSE's ethogram.
        self._interpret.lane_changed.connect(self._browse.set_cluster_lane)
        self._tabs.addTab(self._interpret, INTERPRET)
        self._tabs.addTab(self._audit, AUDIT)
        self._tabs.addTab(self._browse, BROWSE)
        # The §23 numbers for this run's artifact, as a tab of their own rather than a section
        # under every tab: under AUDIT they read as part of the review, and they are not. Live
        # runs only — they read the live artifact, which under a snapshot banner would be numbers
        # for something other than what the page shows.
        self._measures = None
        if run.editable:
            from detail.gui.pages import (
                MeasuresPanel,
            )  # local: pages imports this module

            self._measures = MeasuresPanel(experiment, run.id)
            self._tabs.addTab(top_of_pane(self._measures), MEASURES)
        widgets = [self._interpret, self._audit, self._browse]
        if self._measures is not None:
            widgets.append(self._tabs.widget(3))
        statuses = [
            lambda: _labels_status(run),
            # None: the judged count reads the whole audit log, a known-slow seam that does not
            # belong on the UI thread while the landing page builds.
            None,
            lambda: _recordings_status(run),
            None,
        ]
        self._init_views(
            self._tabs,
            [
                (name, purpose, widget, status)
                for (name, purpose), widget, status in zip(
                    cluster_view_specs(terms, editable=run.editable), widgets, statuses
                )
            ],
        )
        self._tabs.currentChanged.connect(self._on_tab)
        self._audit.node_requested.connect(self._open_node)
        # A judgment on AUDIT changes a cluster's figure: the ring and the card follow it.
        self._audit.changed.connect(self._interpret.refresh_audit)
        layout.addWidget(self._tabs)

    def _on_tab(self, _index: int) -> None:
        """Moving to AUDIT from a selected, labelled cluster audits that cluster's label (#510)."""
        if self._tabs.currentWidget() is not self._audit:
            return
        path = self._interpret.selected_label_path()
        if path and path != self._audit._label:
            self._audit.open_label(path)

    def _open_node(self, node: int) -> None:
        """AUDIT's "Open in INTERPRET": land on the cluster the audit found, to fix it there."""
        self._tabs.setCurrentWidget(self._interpret)
        self._interpret.select(int(node))

    def export_scope(self):
        """What the shell's Export control should offer — asked of the view in front.

        Delegated rather than searched, because two tabs carry a selection and they mean
        different things: INTERPRET has a cluster, BROWSE has a list of instances. Answering
        with whichever was found first would export the other tab's selection.
        """
        current = self._tabs.currentWidget()
        getter = getattr(current, "export_scope", None)
        return getter() if getter is not None else None
