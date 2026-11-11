"""AUDIT on a clustering run: review every clip of each cluster, one cluster at a time (§19, §22).

One screen (review round 11, sidebar option A). **On the left**, a list about 230px wide, two lines
a row (round 12, 4A): each label — a colour swatch, its name in bold and, once every cluster under
it is reviewed, its combined precision; then "62 clips · 1 of 1 clusters reviewed" — with its
clusters indented under it, each "cluster #871 · 62 clips" over a thin bar (how far its scroll has
reached) and "reviewing · 2 removed" / "not reviewed" — or, once reviewed, **✓ Reviewed** in bold
and "97% · 2 removed", with no bar (round 14: a full bar said nothing the mark did not). Clicking a label opens its first cluster nobody
has reviewed; right-clicking a cluster offers **Open in INTERPRET**. **On the right**,
``label_clean.ReviewPane`` for the selected cluster: every clip of it, least typical first, to
remove the wrong ones and finish. A finished cluster's grid is split, "Removed (3)" first and then
"Kept as climb (57)" (round 13, 1B); Reopen review puts back the review order.

**Why every clip** (the user's decision): a random draw of 25 left most of a label's clips unseen,
and a clip nobody watched cannot be removed. So a cluster is reviewed whole and its precision is
a count — the census ``finish_review`` writes — and a label's figure, the clusters' combined by
size, appears only once all of them are reviewed (``audit_figures.complete``). A relabel resets a
label's figures (the backend's ``membership_changed_at`` rule).

**No verdicts.** Whether a precision is good enough is the reader's call (§19).

Every removal and every finish is written in order and off the UI thread (``_write``), so leaving the
page loses nothing. **One copy of what is removed**: ``self._data["taken"]``, the run's take-outs as
last read, updated by each mark the moment it is made, with every mark still in the write queue laid
back over a read that lands before it (``_pending``) — so leaving a cluster and coming back, or a
reload from INTERPRET mid-queue, shows exactly what Finish will record. A clip's edit counts as a
removal here only when it was made from the cluster's label or one above it, the backend's rule
(``api.clip_review._not_its_label``): an edit from before a relabel is not shown as one.

**Reopen review** keeps the old figure on screen only as out of date — "was 97% · reviewing again"
beside the title, "was 97% · 2 removed" with a progress bar on its row — until Finish writes the
new one, even across leaving the cluster.

**Focus**: the grid takes the keyboard only when the user opened the cluster (a click on its row,
Enter, Next cluster, moving here from INTERPRET) and the pane is on screen — never on a reload, which
INTERPRET's label writes start while this tab is hidden. Arrow keys move through the list freely: a
label row opens its first unreviewed cluster on a click or Enter, not when the arrows pass it.

A method that ranks on demand (a medoid order, ``NodeView.ranks_on_demand``) ranks a cluster's clips
off the UI thread with ``run.clips``, as INTERPRET does, and the pane says "ranking this cluster's
clips…" until they arrive. Each change goes on this review's undo history (``_history``) with every clip's
mark before it; ⌘Z restores those marks through the same writes. The earlier one-by-one audit (``audit.csv``) stays readable and is never merged
into these figures: a label's tooltip gives it under "Earlier one-by-one audit".

A query run keeps ``audit_tab.AuditTab``: its behaviors are flat and its selector scopes it.
"""

from datetime import datetime
from datetime import timezone

import numpy as np
from PyQt6.QtCore import QEvent, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFontMetrics, QPalette
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from detail.gui import plot_theme
from detail.gui.busy import run_busy
from detail.gui.grip_splitter import GripSplitter
from detail.gui.label_clean import (
    RANKING,
    ReviewPane,
    ScopeClips,
    clip_key,
    day_of,
    display,
    edit_key,
    tiles_for,
)
from detail.gui.live_montage import ALIGN_CHECKING, context_span
from detail.gui.sizing import emphasis_font, lines
from detail.gui.workers import BackgroundCall
from detail.storage import audit_figures
from detail.storage import behavior_audit as audit
from detail.storage import behavior_output as bo

#: Why a control that would write is dead, by cause.
FROZEN = (
    "This is a snapshot — the labels as they stood when the result left the run. "
    "Go back to the live run to review its clips."
)
FINISHED = "This run is finished. Reopen it to review its clips."
NO_CLIP_EDITS = (
    "This run's method keeps no clip edits: clip review is for behavior (temporal) "
    "clustering runs."
)

#: The list's narrowest default width: room for "cluster #12345" and "not reviewed" side by
#: side. Until it is dragged the list opens as wide as its widest row (`_fit_list_width`), up to
#: `LIST_MAX_LINES`.
LIST_PX = 230
#: The widest the list opens at, in lines, however long a label's name.
LIST_MAX_LINES = 24
#: The narrowest the list drags to: a two-line row's name and figure still fit (round 15).
LIST_MIN_PX = 180
#: Where the list's width is remembered across sessions (`GripSplitter.remember`).
LIST_SPLIT_KEY = "audit_review_list"
#: What a reviewed cluster's row says in bold before its figure.
REVIEWED_MARK = "✓ Reviewed"
#: Where a list row keeps what it is: ``("label", path)`` or ``("cluster", path, node)``.
KEY_ROLE = Qt.ItemDataRole.UserRole
#: Where a list row keeps what it draws (``ReviewListDelegate``).
ROW_ROLE = Qt.ItemDataRole.UserRole + 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_taken(run) -> dict[tuple, dict]:
    """``{clip key: removal record}`` with each record's labels as paths (``from_path`` /
    ``to_path``): the run stores label ids. Off the UI thread."""
    state = run.state()
    resolved = getattr(state, "resolved_paths", None) or {}

    def path_of(label):
        if not label:
            return None
        return resolved.get(label) or state.labels.get(label, label)

    return {
        edit_key(e): {
            **e,
            "from_path": path_of(e.get("from_label")),
            "to_path": path_of(e.get("to_label")),
        }
        for e in run.taken_out()
    }


def label_name(path: str | None) -> str | None:
    """The name the run's calls take for a label path: its last part (names are unique)."""
    return path.rsplit(bo.PATH_SEPARATOR, 1)[-1] if path else None


def own_leaves(view, paths: dict[str, str], node: int) -> np.ndarray:
    """The leaves under ``node`` whose label is ``node``'s own: a labelled node below it keeps
    its leaves (they are its label's, not this one's)."""
    leaves = set(int(x) for x in view.leaves_under(int(node)))
    for other in paths:
        other = int(other)
        if other != int(node) and int(node) in view.ancestors(other):
            leaves -= set(int(x) for x in view.leaves_under(other))
    return np.fromiter(leaves, dtype=np.int64, count=len(leaves))


def scope_rows(view, paths: dict[str, str], node: int, clips=None):
    """``(rows, typicality)``: the clips the clustering gave cluster ``node``, those whose label
    is its own — the backend's review population (``_review_clusters``). ``clips``: the node's
    ``ClipTable`` already ranked (``run.clips``); else the view's, whose typicality is NaN for a
    method that ranks on demand."""
    if clips is None:
        clips = view.clips(int(node))
    if not len(clips):
        return [], np.zeros(0, dtype=np.float64)
    leaf = getattr(clips, "leaf", None)
    if leaf is not None:
        keep = np.isin(np.asarray(leaf), own_leaves(view, paths, int(node)))
        clips = clips.take(np.flatnonzero(keep))
    return clips.rows(), np.asarray(clips.typicality, dtype=np.float64)


def applies(edit: dict, path: str) -> bool:
    """Whether ``edit`` removes its clip from label ``path``: made from ``path`` or a label above
    it — the backend's rule (``api.clip_review._not_its_label``). An edit made from another label,
    before a relabel, is history, not a removal here."""
    source = (edit or {}).get("from_path")
    return bool(source) and audit.under(path, source)


def percent(estimate: dict | None) -> str:
    """ "84%" for a figure, the share rounded to a whole percent."""
    return f"{round(100 * float((estimate or {}).get('precision') or 0.0))}%"


def reviewed(estimate: dict | None) -> bool:
    """Whether every clip in the estimate's scope has been reviewed since its label changed."""
    return audit_figures.complete(estimate)


class ReviewListDelegate(QStyledItemDelegate):
    """How a list row draws, on two lines (review round 12, 4A). A label: swatch, bold name and —
    once every cluster under it is reviewed — its figure on the right; then, muted, "62 clips · 1
    of 1 clusters reviewed". A cluster: its name and "62 clips" on the right; then a thin bar (how
    far its scroll reached) and "reviewing · 2 removed" / "not reviewed" — once reviewed, "✓
    Reviewed" in bold and "100% · 0 removed", with no bar (round 14)."""

    def sizeHint(self, option, index) -> QSize:  # noqa: N802 - Qt virtual
        return QSize(max(LIST_PX, self.row_width(option, index)), lines(2.7))

    def row_width(self, option, index) -> int:
        """The width that draws ``index``'s row with nothing elided: the wider of its two lines,
        plus the margins and the swatch's indent :meth:`paint` uses. A cluster's second line is
        its status beside a bar of at least 12px."""
        row = index.data(ROW_ROLE) or {}
        widget = option.widget
        metrics = option.fontMetrics
        side = row.get("side", "")
        side_w = metrics.horizontalAdvance(side) + 6 if side else 0
        line = row.get("line", "")
        if row.get("kind") == "label":
            bold = QFontMetrics(
                emphasis_font(widget or QApplication.instance(), steps=0)
            )
            first = 16 + bold.horizontalAdvance(row.get("name", "")) + side_w
            second = 16 + metrics.horizontalAdvance(line)
        else:
            first = metrics.horizontalAdvance(row.get("name", "")) + side_w
            second = 12 + metrics.horizontalAdvance(line) + 8
        return max(first, second) + 12 + 2

    def paint(self, painter, option, index) -> None:
        row = index.data(ROW_ROLE) or {}
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(
            QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget
        )
        theme = plot_theme.current()
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        ink = opt.palette.color(
            QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.Text
        )
        muted = ink if selected else QColor(theme.text_secondary)
        rect = opt.rect.adjusted(4, 3, -8, -3)
        first = QRect(rect.x(), rect.y(), rect.width(), rect.height() // 2)
        second = QRect(
            rect.x(), first.bottom() + 1, rect.width(), rect.height() - first.height()
        )
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.save()
        side = row.get("side", "")
        side_w = painter.fontMetrics().horizontalAdvance(side) + 6 if side else 0
        indent = 0
        if row.get("kind") == "label":
            swatch = QRect(first.x(), first.center().y() - 5, 10, 10)
            painter.fillRect(swatch, QColor(row.get("colour") or theme.text))
            indent = 16
            painter.setFont(
                emphasis_font(opt.widget or QApplication.instance(), steps=0)
            )
        name_rect = first.adjusted(indent, 0, -side_w, 0)
        painter.setPen(ink)
        painter.drawText(
            name_rect,
            left,
            painter.fontMetrics().elidedText(
                row.get("name", ""), Qt.TextElideMode.ElideRight, name_rect.width()
            ),
        )
        painter.setFont(opt.font)
        painter.drawText(first, right, side)
        line = row.get("line", "")
        if row.get("kind") == "label":
            painter.setPen(muted)
            room = second.adjusted(indent, 0, 0, 0)
            painter.drawText(
                room,
                left,
                painter.fontMetrics().elidedText(
                    line, Qt.TextElideMode.ElideRight, room.width()
                ),
            )
        else:
            text_w = painter.fontMetrics().horizontalAdvance(line) + 8
            if row.get("state") == "reviewed":
                painter.setFont(
                    emphasis_font(opt.widget or QApplication.instance(), steps=0)
                )
                painter.setPen(ink)
                painter.drawText(second, left, REVIEWED_MARK)
                painter.setFont(opt.font)
            else:
                track = QRect(
                    second.x(),
                    second.center().y() - 1,
                    max(second.right() - text_w - second.x(), 12),
                    3,
                )
                painter.fillRect(track, QColor(theme.grid))
                done = min(max(float(row.get("fraction") or 0.0), 0.0), 1.0)
                if done > 0:
                    painter.fillRect(
                        QRect(track.x(), track.y(), round(track.width() * done), 3),
                        QColor(theme.summary),
                    )
            painter.setPen(ink if row.get("state") == "reviewed" else muted)
            painter.drawText(second, right, line)
        painter.restore()


class LabelCheckTab(QWidget):
    """The label-centred audit of a clustering run: the list, and one cluster's review."""

    changed = pyqtSignal()
    #: "Open in INTERPRET" for a cluster: the workspace switches tabs and selects the node.
    node_requested = pyqtSignal(int)

    PAGE_EMPTY, PAGE_REVIEW = range(2)

    def __init__(self, experiment, run) -> None:
        super().__init__()
        self._experiment = experiment
        self._run = run
        self._data: dict = {}
        #: The label last opened — what moving here from INTERPRET compares against.
        self._label: str | None = None
        self._finished: bool | None = None
        #: The cluster in the pane — ``{path, node, clips, order, edits}`` — or ``None``.
        self._open: dict | None = None
        #: Clusters opened and not finished, by node: how far each scroll reached (0..1).
        self._progress: dict[int, float] = {}
        #: The cluster whose finish is being written; its result shows once it is read back.
        self._finishing: int | None = None
        #: This review's changes, oldest first — what ⌘Z steps back through. Cleared on Finish
        #: and when another cluster opens.
        self._history: list[dict] = []
        #: Writes not yet done, oldest first, and the one being written.
        self._writes: list = []
        self._writer: BackgroundCall | None = None
        #: What runs once every write is done: a read must see the writes made before it.
        self._after_writes: list = []
        #: Marks made and not yet seen by a read: ``(write number, {clip key: edit or None})``,
        #: oldest first — laid over every read that lands before they do (`_render`).
        self._pending: list[tuple[int, dict]] = []
        #: The number of the last write queued, and of the last one done.
        self._queued = 0
        self._landed = 0
        #: Reviewed clusters reopened and not finished again: their old figure is out of date.
        self._reopened: set[int] = set()
        #: Bumped by each cluster opened: a ranking that lands for an earlier one is dropped.
        self._opening = 0
        #: Whether the first read has landed — what a caller waits on.
        self._loaded = False
        self._catalog = None

        root = QVBoxLayout(self)
        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_empty())
        self._pages.addWidget(self._build_review())
        root.addWidget(self._pages, 1)
        self._pages.setCurrentIndex(self.PAGE_EMPTY)
        self.reload()

    # -- construction ----------------------------------------------------------------------
    def _build_empty(self) -> QWidget:
        page = QWidget()
        column = QVBoxLayout(page)
        title = QLabel("Nothing labelled to review yet")
        title.setFont(emphasis_font(title))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(title)
        body = QLabel(
            "Label a node on INTERPRET and its clips can be reviewed here — the unlabelled "
            "remainder makes no claim to check."
        )
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.setWordWrap(True)
        column.addWidget(body)
        column.addStretch(1)
        return page

    def _build_review(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        # A divider, not a fixed column (review round 15): long label names wanted more room,
        # and the width a user drags to is kept across sessions.
        self._split = GripSplitter(Qt.Orientation.Horizontal, key=LIST_SPLIT_KEY)
        self._list = QTreeWidget()
        self._list.setHeaderHidden(True)
        self._list.setColumnCount(1)
        self._list.setMinimumWidth(LIST_MIN_PX)
        self._list.setRootIsDecorated(False)
        self._list.setIndentation(lines(1))
        self._list.setItemDelegate(ReviewListDelegate(self._list))
        self._list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_menu)
        self._list.currentItemChanged.connect(self._on_current)
        # A label row opens its first unreviewed cluster on a click or Enter — never as the
        # arrow keys pass over it, which would keep them from ever moving past it.
        self._list.itemClicked.connect(self._on_chosen)
        self._list.itemActivated.connect(self._on_chosen)
        # macOS's item views emit no activation on Return, so the keys are taken here.
        self._list.installEventFilter(self)
        self._split.addWidget(self._list)
        from detail.gui.montage_export import ExportSource

        self.pane = ReviewPane(
            export=ExportSource(
                clips=self.export_clips,
                experiment_dir=lambda: self._experiment.dir,
                origin=self._origin,
                name=self._export_stem,
                empty=lambda: "Open a cluster in the list to export its clips.",
            )
        )
        self.pane.take_out_requested.connect(self._take_out)
        self.pane.put_back_requested.connect(self._put_back)
        self.pane.undo_requested.connect(self._undo)
        self.pane.finish_requested.connect(self._finish)
        self.pane.next_requested.connect(self._next)
        self.pane.again_requested.connect(self._again)
        self.pane.montage.clip_activated.connect(
            lambda i: QTimer.singleShot(0, lambda: self._play_clip(i))
        )
        self.pane.montage.seen_changed.connect(self._on_seen)
        self.pane.montage.context_changed.connect(
            lambda _s: self._show_tiles(new_source=True)
        )
        self._split.addWidget(self.pane)
        # A wider window widens the clips, not the list.
        self._split.setStretchFactor(0, 0)
        self._split.setStretchFactor(1, 1)
        # The list folds by its grip to a strip reading "Labels", like every side panel.
        self._split.set_foldable(0, "Labels")
        #: Whether the list's width is the user's — remembered from a drag, or dragged now —
        #: rather than fitted to its rows.
        self._list_width_chosen = self._split.remember()
        if not self._list_width_chosen:
            self._split.setSizes([LIST_PX, lines(80)])
        self._split.splitterMoved.connect(self._on_list_dragged)
        self._split.fold_changed.connect(self._on_list_dragged)
        row.addWidget(self._split, 1)
        return page

    # -- reading (off the UI thread) ------------------------------------------------------
    def reload(self) -> None:
        landed = self._landed
        run_busy(
            self,
            self._compute,
            lambda data, n=landed: self._render(data, landed_before=n),
            label="Reading the labels…",
            supersedes="audit",
        )

    def open_label(self, path: str) -> None:
        """Open ``path``'s first cluster nobody has reviewed — what moving here from a cluster
        on INTERPRET does."""
        if not path:
            return
        self._label = path
        if self._loaded:
            self._open_label_row(path, focus=True)

    def _reviewable(self) -> bool:
        return all(
            hasattr(self._run, name)
            for name in ("take_out_clips", "finish_review", "review_by_cluster")
        )

    def _compute(self) -> dict:
        """Every read the page needs, in one off-thread pass."""
        from detail.api import clip_tiles
        from detail.gui.label_panel import label_paths

        run = self._run
        finished = run.editable and run.status == "complete"
        try:
            labels = run.audit_labels()  # off-thread: `_compute`
        except FileNotFoundError:
            return {"labels": [], "finished": finished}
        if not labels:
            return {"labels": [], "finished": finished}
        reviewable = self._reviewable()
        view = run.node_view()  # off-thread
        names = run.node_names()  # off-thread
        paths = label_paths(view, names)
        taken = read_taken(run) if reviewable else {}  # off-thread
        figures = (
            {int(r["node"]): r for r in run.review_by_cluster()}  # off-thread
            if reviewable
            else {}
        )
        theme = plot_theme.current()
        roots = list(dict.fromkeys(p.split(bo.PATH_SEPARATOR)[0] for p in labels))
        clusters: dict[str, list[dict]] = {}
        label_rows = []
        for path in labels:
            label_keys: set = set()
            for node in sorted(int(n) for n, p in paths.items() if p == path):
                rows, _t = scope_rows(view, paths, node)
                keys = {
                    clip_key(
                        r["recording_id"], r["subject"], r["start_t_ns"], r["end_t_ns"]
                    )
                    for r in rows
                }
                label_keys |= keys
                clusters.setdefault(path, []).append(
                    {
                        "node": node,
                        "path": path,
                        "size": len(keys),
                        "keys": frozenset(keys),
                        "estimate": (figures.get(node) or {}).get("estimate") or {},
                        "reviewed_at": (figures.get(node) or {}).get("reviewed_at"),
                    }
                )
            try:
                old = run.audit_precision(path=path) or {}  # off-thread
            except (FileNotFoundError, ValueError):
                old = {}
            label_rows.append(
                {
                    "path": path,
                    "size": len(label_keys),
                    "estimate": (
                        run.review_precision(path) or {} if reviewable else {}
                    ),  # off-thread
                    "old": old,
                    "colour": QColor(
                        theme.identity(roots.index(path.split(bo.PATH_SEPARATOR)[0]))
                    ).name(),
                }
            )
        embeds = {}
        state = run.state()
        for rid in view.recording_ids:
            try:
                embeds[rid] = state.embed_run_for(rid)
            except (KeyError, ValueError):
                continue
        tracking = clip_tiles.tracking_runs(self._experiment.dir, embeds)  # off-thread
        from detail.api.clip_views import pose_notes

        points, align = pose_notes(self._experiment.dir, embeds)  # off-thread
        return {
            "labels": label_rows,
            "clusters": clusters,
            "taken": taken,
            "view": view,
            "paths": paths,
            "names": self._experiment.recording_names(),
            "embeds": embeds,
            "finished": finished,
            "reviewable": reviewable,
            "tracking": tracking,
            "align": align,
            "points": points,
        }

    # -- rendering -----------------------------------------------------------------------
    def _editable(self) -> bool:
        return not self._reason()

    def _reason(self) -> str:
        if not self._run.editable:
            return FROZEN
        if self._finished is None:
            return "Still reading this run…"
        if self._finished:
            return FINISHED
        if not self._data.get("reviewable"):
            return NO_CLIP_EDITS
        return ""

    def _render(self, data: dict, *, landed_before: int | None = None) -> None:
        from detail.api import clip_tiles

        if landed_before is not None:
            # Marks whose write was done before this read began are in it; the rest go back
            # over it, oldest first, so a read never takes back a mark still in the queue.
            self._pending = [p for p in self._pending if p[0] > landed_before]
        taken = dict(data.get("taken") or {})
        for _n, change in self._pending:
            _apply_change(taken, change)
        data = {**data, "taken": taken}
        self._data = data
        self._finished = bool(data.get("finished"))
        self._loaded = True
        if data.get("tracking") is not None:
            self._catalog = clip_tiles.TileCatalog(
                self._experiment, data["tracking"], data.get("embeds")
            )
        self.pane.montage.set_alignment(data.get("align", ALIGN_CHECKING))
        self.pane.montage.set_points(data.get("points", ALIGN_CHECKING))
        if not data.get("labels"):
            self._list.clear()
            self._open = None
            self._pages.setCurrentIndex(self.PAGE_EMPTY)
            return
        self._pages.setCurrentIndex(self.PAGE_REVIEW)
        self.pane.set_editable(self._reason())
        self._fill_list()
        opened = self._open
        if opened is not None and self._cluster(opened["path"], opened["node"]):
            self._select_silently(self._cluster_item(opened["path"], opened["node"]))
            self._refresh_open()
            return
        self._open = None
        path = self._label if self._label_item(self._label or "") else None
        # A read is never the user's step: no focus taken (INTERPRET's writes start these).
        self._open_label_row(path or data["labels"][0]["path"], focus=False)

    def _fill_list(self) -> None:
        """Each label, then its clusters indented under it."""
        lst = self._list
        lst.blockSignals(True)
        lst.clear()
        for row in self._data.get("labels", []):
            item = QTreeWidgetItem([display(row["path"])])
            item.setData(0, KEY_ROLE, ("label", row["path"]))
            item.setToolTip(0, self._label_tip(row))
            lst.addTopLevelItem(item)
            for cluster in self._data.get("clusters", {}).get(row["path"], []):
                child = QTreeWidgetItem([f"cluster #{cluster['node']}"])
                child.setData(
                    0, KEY_ROLE, ("cluster", row["path"], int(cluster["node"]))
                )
                child.setToolTip(0, self._cluster_tip(cluster))
                item.addChild(child)
            item.setExpanded(True)
        self._paint_rows()
        lst.blockSignals(False)
        self._fit_list_width()

    def _on_list_dragged(self, *_args) -> None:
        self._list_width_chosen = True

    def _fit_list_width(self, *, grow_only: bool = False) -> None:
        """Open the list as wide as its widest row, between `LIST_PX` and `LIST_MAX_LINES`
        lines — until the user has dragged it, which `GripSplitter` remembers.

        A fixed 230px cut every label's second line at 1440x847: "182 clips · 0 of 3 clusters
        review…". Rows still elide, with the whole text on hover, past the cap or once dragged
        narrower.
        """
        if self._list_width_chosen or self._split.is_folded(0):
            return
        lst = self._list
        delegate = lst.itemDelegate()
        option = QStyleOptionViewItem()
        option.initFrom(lst)
        widest = 0
        walk = QTreeWidgetItemIterator(lst)
        while walk.value() is not None:
            item = walk.value()
            index = lst.indexFromItem(item)
            depth = 0 if item.parent() is None else 1
            widest = max(
                widest, delegate.row_width(option, index) + depth * lst.indentation()
            )
            walk += 1
        chrome = 2 * lst.frameWidth() + lst.verticalScrollBar().sizeHint().width()
        width = min(max(widest + chrome, LIST_PX), lines(LIST_MAX_LINES))
        sizes = self._split.sizes()
        total = sum(sizes)
        # A row that grew as it changed widens the list; it never jumps narrower mid-review.
        if total > 0 and sizes[0] != width and not (grow_only and width < sizes[0]):
            self._split.setSizes([width, max(total - width, 1)])

    def _under(self, path: str) -> list[dict]:
        """Every cluster at or under label ``path`` — what its figure combines."""
        return [
            c
            for p, cs in self._data.get("clusters", {}).items()
            if audit.under(p, path)
            for c in cs
        ]

    def _paint_rows(self) -> None:
        """What each row draws, from the data and the pane's progress."""
        walk = QTreeWidgetItemIterator(self._list)
        while walk.value() is not None:
            item = walk.value()
            key = item.data(0, KEY_ROLE)
            if key[0] == "label":
                row = self._label_row(key[1]) or {}
                item.setToolTip(0, self._label_tip(row) if row else "")
                estimate = row.get("estimate") or {}
                under = self._under(key[1])
                done = sum(1 for c in under if reviewed(c.get("estimate")))
                clips = sum(int(c.get("size") or 0) for c in under)
                item.setData(
                    0,
                    ROW_ROLE,
                    {
                        "kind": "label",
                        "name": display(key[1]),
                        "colour": row.get("colour"),
                        "side": percent(estimate) if reviewed(estimate) else "",
                        "line": f"{clips:,} clips · {done} of {len(under)} clusters "
                        "reviewed",
                    },
                )
            else:
                cluster = self._cluster(key[1], key[2]) or {}
                if cluster:
                    item.setToolTip(0, self._cluster_tip(cluster))
                item.setData(0, ROW_ROLE, self._cluster_look(cluster))
            walk += 1
        self._fit_list_width(grow_only=True)
        self._list.viewport().update()

    def _cluster_look(self, cluster: dict) -> dict:
        node = int(cluster.get("node", -1))
        estimate = cluster.get("estimate") or {}
        opened = self._open
        is_open = opened is not None and opened["node"] == node
        out = len(opened["edits"]) if is_open else self._taken_in(cluster)
        if node in self._reopened and reviewed(estimate):
            state, fraction = "reviewing", self._progress.get(node, 0.0)
            # Short: the row is narrow. The pane says "reviewing again" beside the title.
            status = f"was {percent(estimate)} · {out:,} removed"
        elif reviewed(estimate) and not (is_open and not self.pane.is_finished()):
            state, fraction = "reviewed", 1.0
            status = f"{percent(estimate)} · {out:,} removed"
        elif node in self._progress:
            state, fraction = "reviewing", self._progress[node]
            status = f"reviewing · {out:,} removed"
        else:
            state, fraction, status = "not", 0.0, "not reviewed"
        return {
            "kind": "cluster",
            "name": f"cluster #{node}",
            "state": state,
            "side": f"{int(cluster.get('size') or 0):,} clips",
            "line": status,
            "fraction": fraction,
        }

    @staticmethod
    def _was(estimate: dict) -> str:
        """A reopened cluster's old figure, said as out of date."""
        return f"was {percent(estimate)} · reviewing again"

    def _taken_in(self, cluster: dict) -> int:
        """How many of ``cluster``'s clips are removed from its label now — the one copy,
        under the backend's rule (`applies`)."""
        taken = self._data.get("taken", {})
        path = cluster.get("path", "")
        return sum(
            1 for k in cluster.get("keys", ()) if k in taken and applies(taken[k], path)
        )

    def _label_tip(self, row: dict) -> str:
        path = row["path"]
        under = self._under(path)
        done = sum(1 for c in under if reviewed(c.get("estimate")))
        parts = [display(path)]
        estimate = row.get("estimate") or {}
        if reviewed(estimate):
            parts.append(
                f"precision {percent(estimate)}: every clip of its {len(under)} "
                f"cluster{'s' if len(under) != 1 else ''} reviewed, each weighted by its size"
            )
        else:
            parts.append(
                f"{done} of {len(under)} cluster{'s' if len(under) != 1 else ''} reviewed — "
                "its precision shows once every one is"
            )
        own = self._data.get("clusters", {}).get(path, [])
        taken = sum(self._taken_in(c) for c in under)
        size = sum(int(c.get("size") or 0) for c in under)
        # The same clusters as the row's "N clips": its own and its sub-labels'.
        parts.append(f"{taken:,} of {size:,} clips removed")
        if len(under) > len(own):
            parts.append("Its sub-labels' clusters are counted with its own.")
        old = row.get("old") or {}
        if old.get("n"):
            parts.append(f"Earlier one-by-one audit: {audit_figures.line(old)}")
        return "\n".join(parts)

    def _cluster_tip(self, cluster: dict) -> str:
        estimate = cluster.get("estimate") or {}
        first = (
            f"cluster #{cluster['node']}: {cluster['size']} clips of "
            f"{display(cluster['path'])}, {self._taken_in(cluster)} removed"
        )
        if int(cluster["node"]) in self._reopened and reviewed(estimate):
            second = (
                f"Reopened: precision {percent(estimate)} until it is finished again"
            )
        elif reviewed(estimate):
            second = f"Reviewed: precision {percent(estimate)}"
        else:
            second = "Not reviewed yet: every clip of it needs a look."
        return "\n".join([first, second, "Right-click to open it on INTERPRET."])

    # -- finding rows --------------------------------------------------------------------
    def _label_row(self, path: str) -> dict | None:
        return next(
            (r for r in self._data.get("labels", []) if r["path"] == path), None
        )

    def _cluster(self, path: str, node: int) -> dict | None:
        return next(
            (
                c
                for c in self._data.get("clusters", {}).get(path, [])
                if int(c["node"]) == int(node)
            ),
            None,
        )

    def _ordered_clusters(self) -> list[dict]:
        """Every cluster in list order."""
        return [
            c
            for row in self._data.get("labels", [])
            for c in self._data.get("clusters", {}).get(row["path"], [])
        ]

    def _items(self):
        walk = QTreeWidgetItemIterator(self._list)
        while walk.value() is not None:
            yield walk.value()
            walk += 1

    def _label_item(self, path: str) -> QTreeWidgetItem | None:
        return next(
            (i for i in self._items() if tuple(i.data(0, KEY_ROLE)) == ("label", path)),
            None,
        )

    def _cluster_item(self, path: str, node: int) -> QTreeWidgetItem | None:
        key = ("cluster", path, int(node))
        return next(
            (i for i in self._items() if tuple(i.data(0, KEY_ROLE)) == key), None
        )

    def _select_silently(self, item: QTreeWidgetItem | None) -> None:
        self._list.blockSignals(True)
        self._list.setCurrentItem(item)
        self._list.blockSignals(False)

    # -- the list --------------------------------------------------------------------------
    def _on_current(self, current: QTreeWidgetItem | None, _previous) -> None:
        """The arrow keys (or a click's press) moved the list: a cluster opens, keeping the
        keyboard in the list; a label row is only passed over."""
        if current is None:
            return
        key = current.data(0, KEY_ROLE)
        self._label = key[1]
        if key[0] == "cluster":
            self.open_cluster(key[1], key[2])

    def _on_chosen(self, item: QTreeWidgetItem | None, _column: int = 0) -> None:
        """A row clicked or Entered: a label opens its first unreviewed cluster; either way the
        clips take the keyboard."""
        if item is None:
            return
        key = item.data(0, KEY_ROLE)
        if key[0] == "label":
            self._open_label_row(key[1], focus=True)
        else:
            self._take_focus()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt virtual
        """Return, Enter or Space on a list row chooses it, as a click does."""
        if (
            watched is self._list
            and event.type() == QEvent.Type.KeyPress
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space)
            and self._list.currentItem() is not None
        ):
            self._on_chosen(self._list.currentItem())
            return True
        return super().eventFilter(watched, event)

    def _open_label_row(self, path: str, *, focus: bool) -> None:
        """Open label ``path``'s first cluster nobody has reviewed (its first, if all are)."""
        self._label = path
        clusters = self._data.get("clusters", {}).get(path, [])
        if not clusters:
            item = self._label_item(path)
            if item is not None:
                self._select_silently(item)
            return
        first = next(
            (c for c in clusters if not reviewed(c.get("estimate"))), clusters[0]
        )
        self.open_cluster(path, first["node"], focus=focus)

    def _take_focus(self) -> None:
        """Give the clips the keyboard — only while they are on screen: a hidden tab that takes
        focus leaves the visible page with none."""
        if self.pane.isVisible():
            self.pane.montage.canvas.setFocus()

    def _on_menu(self, point) -> None:
        item = self._list.itemAt(point)
        if item is None:
            return
        key = item.data(0, KEY_ROLE)
        if key[0] != "cluster":
            return
        menu = QMenu(self._list)
        menu.addAction(
            "Open in INTERPRET", lambda n=key[2]: self.node_requested.emit(int(n))
        )
        from detail.gui.menus import open_once

        open_once(menu, self._list.viewport().mapToGlobal(point), wait=False)

    # -- the review pane -------------------------------------------------------------------
    def open_cluster(self, path: str, node: int, *, focus: bool = False) -> None:
        """Show cluster ``node`` of label ``path`` in the pane: its result once reviewed (and
        not reopened), else every clip to review. ``focus``: the user opened it, so the clips
        take the keyboard (only if they are on screen)."""
        data = self._data
        cluster = self._cluster(path, node)
        if cluster is None or data.get("view") is None:
            return
        self._opening += 1
        opening = self._opening
        view = data["view"]
        ranks = bool(getattr(view, "ranks_on_demand", False))
        self._open = {
            "path": path,
            "node": int(node),
            "clips": ScopeClips([], np.zeros(0), data.get("names", {})),
            "order": [],
            "edits": {},
            "display": [],
        }
        self._select_silently(self._cluster_item(path, node))
        self._forget_history()
        row = self._label_row(path) or {}
        self.pane.show_cluster(
            int(node), path, QColor(row["colour"]) if row.get("colour") else None
        )
        self.pane.set_targets(self._targets(path))
        self.pane.set_editable(self._reason())
        estimate = cluster.get("estimate")
        if reviewed(estimate) and int(node) not in self._reopened:
            self.pane.set_was("")
            self._show_result(cluster)
        else:
            self.pane.set_was(
                self._was(estimate) if int(node) in self._reopened else ""
            )
            self.pane.start_review()
            self._progress.setdefault(int(node), 0.0)
        if focus:
            self._take_focus()
        if not ranks:
            self.pane.set_waiting("")
            self._with_clips(scope_rows(view, data["paths"], int(node)))
            return
        # Ranked from the fit on first ask: slow on a large cluster, so off the UI thread.
        self.pane.set_waiting(RANKING)
        self.pane.montage.set_clips([])
        self._paint_rows()
        run, paths = self._run, data["paths"]
        reader = BackgroundCall(
            lambda: scope_rows(view, paths, int(node), run.clips(int(node))),
            receiver=self,
        )
        reader.done.connect(
            lambda ranked: opening == self._opening and self._with_clips(ranked)
        )
        reader.failed.connect(
            lambda exc: opening == self._opening
            and self.pane.set_waiting(f"the clips could not be ranked: {exc}")
        )
        reader.finished.connect(reader.deleteLater)
        reader.start()

    def _with_clips(self, ranked) -> None:
        """The open cluster's clips have arrived (ranked): show them with their marks."""
        opened = self._open
        rows, typicality = ranked
        clips = ScopeClips(rows, typicality, self._data.get("names", {}))
        opened["clips"] = clips
        opened["order"] = clips.least_typical_first()
        opened["edits"] = self._edits_for(opened["path"], clips)
        opened.pop("display", None)
        self.pane.set_waiting("")
        self._show_tiles(scroll_to_top=True)
        self._paint_rows()

    def _edits_for(self, path: str, clips: ScopeClips) -> dict:
        """``{clip key: edit}`` for the clips removed from ``path`` now (`applies`)."""
        taken = self._data.get("taken", {})
        out = {}
        for i in range(len(clips.rows)):
            key = clips.key(i)
            edit = taken.get(key)
            if edit is not None and applies(edit, path):
                out[key] = edit
        return out

    def _targets(self, path: str) -> list[str]:
        """The labels a clip may move to, most clips first: every label but ``path``."""
        rows = sorted(
            (r for r in self._data.get("labels", []) if r["path"] != path),
            key=lambda r: -int(r.get("size") or 0),
        )
        return [r["path"] for r in rows]

    def _display(self) -> tuple[list[int], list[tuple[str, int]] | None]:
        """The open cluster's clips in the order shown, and the grid's sections: the review
        order while reviewing; once finished, the removed clips first and then the kept ones,
        each under its heading (round 13, 1B)."""
        opened = self._open
        order = [int(i) for i in opened["order"]]
        if not self.pane.is_finished():
            return order, None
        clips: ScopeClips = opened["clips"]
        edits = opened["edits"]
        removed = [i for i in order if clips.key(i) in edits]
        kept = [i for i in order if clips.key(i) not in edits]
        name = display(opened["path"])
        sections = [(f"Removed ({len(removed):,})", len(removed))] if removed else []
        sections.append((f"Kept as {name} ({len(kept):,})", len(kept)))
        return removed + kept, sections

    def _show_tiles(
        self, *, scroll_to_top: bool = False, new_source: bool = False
    ) -> None:
        """Fill the montage from ``self._open``: faded where marked; the scroll kept unless asked
        to start at the top, or the clips' order changed (a cluster finished, or reviewed
        again). ``new_source``: the tiles' pictures are read afresh (the context changed).
        """
        from detail.api import clip_tiles

        opened = self._open
        if opened is None:
            return
        clips: ScopeClips = opened["clips"]
        order, sections = self._display()
        if order != opened.get("display"):
            scroll_to_top = True
        opened["display"] = order
        montage = self.pane.montage
        context = montage.context_s()
        montage.set_clips(
            tiles_for(clips, order, opened["edits"], context),
            keep_scroll=not scroll_to_top,
            sections=sections,
        )
        if (scroll_to_top or new_source) and self._catalog is not None:
            tiles = []
            for i in order:
                row = clips.rows[int(i)]
                start, end = context_span(row["start_t_ns"], row["end_t_ns"], context)
                tiles.append(
                    clip_tiles.TileClip(
                        row["recording_id"], int(row["subject"]), start, end
                    )
                )
            montage.set_source(clip_tiles.ClipTiles(self._catalog, tiles))
        montage.set_hover_text(self._hover)
        item = self._cluster_item(opened["path"], opened["node"])
        if item is not None:
            item.setData(
                0,
                ROW_ROLE,
                self._cluster_look(self._cluster(opened["path"], opened["node"]) or {}),
            )
        self.pane.refresh()

    def _refresh_open(self) -> None:
        """After a read: the open cluster's marks as the run has them, the scroll kept, and its
        result if a finish was waiting on this read."""
        opened = self._open
        opened["edits"] = self._edits_for(opened["path"], opened["clips"])
        cluster = self._cluster(opened["path"], opened["node"]) or {}
        if self._finishing == opened["node"] and reviewed(cluster.get("estimate")):
            self._finishing = None
            self._progress.pop(opened["node"], None)
            self._reopened.discard(opened["node"])
            self.pane.set_was("")
            self._show_result(cluster)
        self._show_tiles()
        self._paint_rows()

    def _hover(self, index: int) -> str:
        opened = self._open
        if opened is None or not 0 <= index < len(opened["display"]):
            return ""
        i = int(opened["display"][index])
        clips: ScopeClips = opened["clips"]
        row = clips.rows[i]
        seconds = (int(row["end_t_ns"]) - int(row["start_t_ns"])) / 1e9
        return f"#{index + 1} · {clips.caption(i)} · {seconds:.1f} s"

    # -- what Export ▾ writes ------------------------------------------------------------
    def export_clips(self) -> list:
        """The open cluster's clips in the order shown as ``api.clip_montage.MontageClip`` —
        what **Export ▾** writes (``montage_export.ExportSource``), each its own span, ranked
        by its place on the grid."""
        from detail.api import clip_montage as montage_api

        opened = self._open
        if opened is None or "display" not in opened:
            return []
        clips: ScopeClips = opened["clips"]
        embeds = self._data.get("embeds", {})
        out = []
        for rank, i in enumerate(opened["display"], start=1):
            row = clips.rows[int(i)]
            rid = row["recording_id"]
            out.append(
                montage_api.MontageClip(
                    rank=rank,
                    recording_id=rid,
                    recording_name=clips.names.get(rid, rid),
                    subject=int(row["subject"]),
                    embed_run_id=embeds.get(rid, ""),
                    start_t_ns=int(row["start_t_ns"]),
                    end_t_ns=int(row["end_t_ns"]),
                )
            )
        return out

    def _export_stem(self) -> str:
        opened = self._open or {}
        name = label_name(opened.get("path")) or "cluster"
        return f"{name}_cluster_{opened.get('node', '')}"

    def _origin(self):
        from detail.api.clip_montage import MontageOrigin

        opened = self._open or {}
        return MontageOrigin(
            run_id=str(self._run.id),
            node=int(opened.get("node", -1)),
            name=label_name(opened.get("path")),
        )

    def _on_seen(self, seen: int) -> None:
        opened = self._open
        total = len(opened["order"]) if opened is not None else 0
        if not total or opened["node"] not in self._progress:
            return
        self._progress[opened["node"]] = seen / total
        item = self._cluster_item(opened["path"], opened["node"])
        if item is not None:
            item.setData(
                0,
                ROW_ROLE,
                self._cluster_look(self._cluster(opened["path"], opened["node"]) or {}),
            )

    def _result_text(self, cluster: dict) -> str:
        """ "57 of 60 are climb · 3 removed": the reviewed box's counts."""
        estimate = cluster.get("estimate") or {}
        n = int(estimate.get("n") or 0)
        yes = int(estimate.get("n_confirmed") or 0)
        name = display(cluster["path"])
        return f"{yes:,} of {n:,} are {name} · {n - yes:,} removed"

    def _show_result(self, cluster: dict) -> None:
        """The reviewed box for ``cluster``; the caller's ``_show_tiles`` then splits the grid."""
        estimate = cluster.get("estimate") or {}
        self.pane.show_result(
            percent(estimate),
            self._result_text(cluster),
            self._next_node(),
            removed=int(estimate.get("n") or 0) - int(estimate.get("n_confirmed") or 0),
            when=day_of(cluster.get("reviewed_at")),
        )

    def _next_node(self) -> int | None:
        """The next cluster nobody has reviewed, in list order after the open one (wrapping)."""
        ordered = self._ordered_clusters()
        opened = self._open
        at = next(
            (
                i
                for i, c in enumerate(ordered)
                if opened is not None and int(c["node"]) == opened["node"]
            ),
            -1,
        )
        for c in ordered[at + 1 :] + ordered[: max(at, 0)]:
            if not reviewed(c.get("estimate")) and (
                opened is None or int(c["node"]) != opened["node"]
            ):
                return int(c["node"])
        return None

    def _next(self) -> None:
        node = self._next_node()
        cluster = next(
            (c for c in self._ordered_clusters() if int(c["node"]) == node), None
        )
        if cluster is not None:
            self._label = cluster["path"]
            self.open_cluster(cluster["path"], node, focus=True)

    def _again(self) -> None:
        if self._open is None or not self._editable():
            return
        node = self._open["node"]
        self._progress[node] = 0.0
        cluster = self._cluster(self._open["path"], node) or {}
        if reviewed(cluster.get("estimate")):
            self._reopened.add(node)
            self.pane.set_was(self._was(cluster.get("estimate")))
        self.pane.start_review()
        self._show_tiles(scroll_to_top=True)
        self._paint_rows()

    def _rows_at(self, indices) -> list[dict]:
        opened = self._open
        return [opened["clips"].rows[int(opened["display"][int(i)])] for i in indices]

    def _key_of(self, row: dict) -> tuple:
        return clip_key(
            row["recording_id"], row["subject"], row["start_t_ns"], row["end_t_ns"]
        )

    def _take_out(self, indices: list, to_label) -> None:
        """Remove the clips at display ``indices`` from the label (or move them to
        ``to_label``): faded at once, written in order off the UI thread, and undoable.
        """
        opened = self._open
        if opened is None or not self._editable() or not indices:
            return
        rows = self._rows_at(indices)
        name = display(opened["path"])
        text = (
            f"Moved {len(rows):,} to {display(to_label)}"
            if to_label
            else f"Removed {len(rows):,} from {name}"
        )
        self._remember(rows, text)
        self._set_marks(rows, to_label)

    def _put_back(self, indices: list) -> None:
        opened = self._open
        if opened is None or not self._editable() or not indices:
            return
        rows = self._rows_at(indices)
        self._remember(rows, f"Put back {len(rows):,} in {display(opened['path'])}")
        self._set_marks(rows, None, back=True)

    def _remember(self, rows: list[dict], text: str) -> None:
        """Push a change onto this review's undo history: each clip's mark before it."""
        edits = self._open["edits"]
        self._history.append(
            {
                "text": text,
                "rows": list(rows),
                "before": {self._key_of(r): edits.get(self._key_of(r)) for r in rows},
            }
        )
        self.pane.set_undo(text)

    def _set_marks(self, rows: list[dict], to_label, *, back: bool = False) -> None:
        """Mark ``rows`` (``back``: put them back) at once on screen, then write it in order."""
        opened = self._open
        path = opened["path"]
        change = {
            self._key_of(row): (
                None if back else _edit(self._key_of(row), path, to_label)
            )
            for row in rows
        }
        # The one copy first; what the pane shows is read from it.
        _apply_change(self._data.setdefault("taken", {}), change)
        opened["edits"] = self._edits_for(path, opened["clips"])
        self._show_tiles()
        self._paint_rows()
        run = self._run
        if back:
            self._write(
                lambda r=rows: run.put_back_clips(r), "Clips not put back", change
            )
        else:
            self._write(
                lambda r=rows, p=label_name(path), t=label_name(
                    to_label
                ): run.take_out_clips(r, p, t),
                "Clips not removed",
                change,
            )

    def _undo(self) -> None:
        """Reverse the last change of this review: every clip back to the mark it had, through
        the same ordered writes."""
        opened = self._open
        if opened is None or not self._editable() or not self._history:
            return
        change = self._history.pop()
        unmarked, by_target = [], {}
        for row in change["rows"]:
            before = change["before"][self._key_of(row)]
            if before is None:
                unmarked.append(row)
            else:
                by_target.setdefault(before.get("to_path"), []).append(row)
        if unmarked:
            self._set_marks(unmarked, None, back=True)
        for target, rows in by_target.items():
            self._set_marks(rows, target)
        self.pane.set_undo(self._history[-1]["text"] if self._history else None)

    def _forget_history(self) -> None:
        self._history = []
        self.pane.set_undo(None)

    def _finish(self) -> None:
        """Write the census — after every mark already queued — then read the figure back."""
        opened = self._open
        if opened is None or not self._editable() or not self.pane.reached_end():
            return
        node = opened["node"]
        self._finishing = node
        self._forget_history()
        self.pane.show_saving()
        run = self._run
        self._write(lambda n=node: run.finish_review(n), "Review not saved")
        self._when_written(self.reload)

    def _play_clip(self, index: int) -> None:
        """A tile double-clicked (or Space): the clip in the clip window, from the source video,
        with previous / next through the clips in the order shown."""
        from detail.gui import clip_popup
        from detail.gui.behavior_query import ClipSpec

        opened = self._open
        if opened is None or not 0 <= index < len(opened["display"]):
            return
        clips: ScopeClips = opened["clips"]
        embeds = self._data.get("embeds", {})
        sequence = []
        for rank, i in enumerate(opened["display"], start=1):
            row = clips.rows[int(i)]
            rid = row["recording_id"]
            if rid not in embeds:
                return
            sequence.append(
                ClipSpec(
                    rid,
                    int(row["subject"]),
                    embeds[rid],
                    int(row["start_t_ns"]),
                    int(row["end_t_ns"]),
                    f"#{rank} · {clips.caption(int(i))}",
                )
            )
        clip_popup.open_clip_window(
            self,
            self._experiment,
            sequence[index],
            sequence=sequence,
            index=index,
            context_s=1,
            order_note=(
                "Removed first, then kept"
                if self.pane.is_finished()
                else "Least typical first"
            ),
        )

    # -- writing (off the UI thread, in order) ---------------------------------------------
    def _write(
        self, write, what: str = "Not saved", change: dict | None = None
    ) -> None:
        """Queue one write. Writes run one at a time, in the order made, so a put-back always
        lands after the take-out it undoes and a finish after every mark before it. ``change``:
        the marks it writes, laid over any read that lands before it does."""
        self._queued += 1
        if change:
            self._pending.append((self._queued, change))
        self._writes.append((write, what, self._queued))
        self._write_next()

    def _write_next(self) -> None:
        if self._writer is not None:
            return
        if not self._writes:
            waiting, self._after_writes = self._after_writes, []
            for then in waiting:
                then()
            return
        write, what, number = self._writes.pop(0)
        writer = BackgroundCall(write, receiver=self)
        self._writer = writer

        def done(_result) -> None:
            self._writer = None
            self._landed = number
            if not self._writes:
                self.changed.emit()
            self._write_next()

        def failed(exc) -> None:
            self._writer = None
            self._writes.clear()
            self._after_writes.clear()
            # Nothing queued will be written: the next read is the run as it stands.
            self._pending.clear()
            self._finishing = None
            QMessageBox.critical(self, what, str(exc))
            self._open = None
            self.reload()

        writer.done.connect(done)
        writer.failed.connect(failed)
        writer.finished.connect(writer.deleteLater)
        writer.start()

    def _when_written(self, then) -> None:
        """Run ``then`` once every queued write is done — at once if none are waiting."""
        self._after_writes.append(then)
        self._write_next()

    def writes_pending(self) -> bool:
        """Whether a change made is not yet on disk."""
        return self._writer is not None or bool(self._writes)


def _apply_change(taken: dict, change: dict) -> None:
    """Lay ``{clip key: edit or None}`` over ``taken``: ``None`` puts the clip back."""
    for key, edit in change.items():
        if edit is None:
            taken.pop(key, None)
        else:
            taken[key] = edit


def _edit(key: tuple, path: str, to_label) -> dict:
    """A removal record as the page holds it until the run's own is read back."""
    return {
        "recording_id": key[0],
        "subject": key[1],
        "t_ns": key[2],
        "end_t_ns": key[3],
        "from_label": path,
        "to_label": to_label,
        "from_path": path,
        "to_path": to_label,
        "at": _now(),
    }
