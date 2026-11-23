"""BROWSE on a clustering run: the labelled dataset, as it stands (§21/§22).

INTERPRET is where a label is decided, one node at a time, from a cached sample. This is the
other half: every instance the labels yield, exactly as the run writes them, filtered down to
the slice someone wants to look at — a label, some recordings, a metadata group — with a
clickable ethogram over it and a sortable list beside the player.

**Filters here, charts there.** The dimensions are the node card's (which videos, which group),
but INTERPRET *reads* them as charts about one node and this *narrows* by them across the whole
labelled set. Charting them again here would be the same picture twice.

**Labelled time only.** The unlabelled remainder is the run's own bucket, not something anyone
built, and a list mostly made of it buries the dataset this tab exists to show.

**The cluster selected on INTERPRET** is a thin lane under each recording's strip — its time,
labelled or not (:meth:`LabelBrowseTab.set_cluster_lane`). Click a mark to play that stretch.

A leaf: it imports the player and the shared widgets, and nothing from ``gui.pages`` or
``gui.modules``: importing a hub from a leaf is how this package's import cycles formed
(``tests/gui/test_import_cycles.py``).
"""

import logging

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableView,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from detail.core import run_selection
from detail.core.progress import format_duration
from detail.gui import plot_theme
from detail.gui.behavior_query import ClipSpec
from detail.gui.busy import run_busy
from detail.gui.cluster_strip import EthogramStrip, cluster_colors, ruled
from detail.gui.collapse import CollapsibleBand
from detail.gui.grip_splitter import GripSplitter
from detail.gui.label_panel import path_rows, root_slots, row_path
from detail.gui.sizing import emphasis_font, lines, set_available
from detail.gui.status_color import MUTED, set_label_color
from detail.gui.tables import (
    BROWSE_ROWS,
    SortCell,
    enable_sorting,
    give_slack_to,
    fit_table_height,
    make_columns_resizable,
    num_cell,
    top_of_pane,
)
from detail.gui.workers import BackgroundCall
from detail.storage import behavior_audit as audit
from detail.storage import behavior_output as bo

#: How many rows the list builds. Not a display preference: a run with tens of thousands
#: of instances would be seconds of frozen window. The ethogram has no cap — it
#: draws every filtered bout — and a band past the cap still plays when clicked.
ROW_CAP = 2000

#: Recordings the ethogram shows before it scrolls.
_ETHOGRAM_ROWS = 6

#: The narrowest the player beside the list is fitted to, in lines: the list gets its columns'
#: width first, up to leaving the player this much (a 340px picture at the system font).
_PLAYER_MIN_LINES = 20

#: What a recording with no value in the chosen metadata column is filed under.
NO_VALUE = "(no value)"

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# the dataset, without Qt
# --------------------------------------------------------------------------- #


def _lane_runs(view, node: int, name: str) -> dict[str, list[dict]]:
    """``{recording_id: [run, ...]}``: node ``node``'s stretches as fitted, each named ``name``.
    Walks every leaf run — off the UI thread (``LabelBrowseTab._read_lane``)."""
    runs: dict[str, list[dict]] = {}
    named = {str(int(node)): name}
    fitted = view.labelled_runs(named, apply_edits=False)  # off-thread: `_read_lane`
    for run in fitted:
        runs.setdefault(run["recording_id"], []).append({**run, "label": name})
    return runs


def path_of(row) -> str:
    """The label path a bout reports under — its ``label_path``, or its flat label."""
    return row.label_path or row.label


def labelled_bouts(instances) -> list:
    """One entry per labelled bout, in recording and time order — the dataset this tab shows.

    Collapsed through ``bouts`` first (``instances.csv`` is one row per ``(bout, label)``), and
    without the unlabelled remainder, whose ``label`` is empty.
    """
    rows = [row for row in bo.bouts(instances) if row.label]
    rows.sort(key=lambda r: (r.recording_id, r.subject or 0, r.t_ns))
    return rows


def filter_bouts(
    rows,
    *,
    paths: set[str] | None = None,
    recordings: set[str] | None = None,
    meta_column: str | None = None,
    meta_values: set[str] | None = None,
    metadata: dict | None = None,
) -> list:
    """The bouts that pass every filter. ``None`` for a filter means "no filter".

    ``paths`` is exact: a parent label's own bouts are its path, and its sub-labels' are theirs,
    so ticking "rearing" and not "supported" shows rearing's other bouts only.
    """
    return [
        row
        for row in rows
        if (paths is None or path_of(row) in paths)
        and recording_passes(
            row.recording_id,
            recordings=recordings,
            meta_column=meta_column,
            meta_values=meta_values,
            metadata=metadata,
        )
    ]


def recording_passes(
    recording_id: str,
    *,
    recordings: set[str] | None = None,
    meta_column: str | None = None,
    meta_values: set[str] | None = None,
    metadata: dict | None = None,
) -> bool:
    """Whether a recording passes the recording and metadata filters — which strips the
    ethogram draws, including a recording with no labelled time (an empty strip is a finding).
    """
    if recordings is not None and recording_id not in recordings:
        return False
    if meta_column is None or meta_values is None:
        return True
    value = ((metadata or {}).get(recording_id) or {}).get(meta_column) or NO_VALUE
    return value in meta_values


def label_table(
    rows, periods: dict[str, int]
) -> list[tuple[str, int, str, int, int, str]]:
    """``[(text, depth, root, bouts, time_ns, path)]`` — the label filter's rows, nested like
    INTERPRET's labels panel (:func:`~detail.gui.label_panel.path_rows`)."""
    time_by_path: dict[str, int] = {}
    count_by_path: dict[str, int] = {}
    for row in rows:
        path = path_of(row)
        time_by_path[path] = time_by_path.get(path, 0) + bo.instance_ns(row, periods)
        count_by_path[path] = count_by_path.get(path, 0) + 1
    out = []
    for text, depth, root, time_ns in path_rows(time_by_path):
        path = row_path(text, depth, root)
        if depth == 0:
            bouts = sum(
                n
                for p, n in count_by_path.items()
                if p.split(bo.PATH_SEPARATOR)[0] == root
            )
        else:
            bouts = count_by_path.get(path, 0)
        out.append((text, depth, root, bouts, time_ns, path))
    return out


def recording_spans(coverage) -> dict[str, tuple[int, int]]:
    """``{recording_id: (first_ns, last_ns)}`` over every subject — the ethogram's x axis."""
    spans: dict[str, tuple[int, int]] = {}
    for row in coverage:
        first, last = spans.get(row.recording_id, (row.start_t_ns, row.end_t_ns))
        spans[row.recording_id] = (
            min(first, row.start_t_ns),
            max(last, row.end_t_ns),
        )
    return spans


def _clock(t_ns: int) -> str:
    """Where in the recording something happens — a position, always ``h:mm:ss``."""
    total = int(t_ns // 1_000_000_000)
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _display(path: str) -> str:
    return path.replace(bo.PATH_SEPARATOR, audit.PATH_DISPLAY)


# --------------------------------------------------------------------------- #
# the tab
# --------------------------------------------------------------------------- #
class LabelBrowseTab(QWidget):
    """Every labelled instance: filter it, see where it falls, watch any of it."""

    def __init__(self, run, experiment, terms) -> None:
        from detail.gui.behavior_query import (
            _ClipPlayer,
        )  # local: avoids an import cycle

        super().__init__()
        self._run = run
        self._experiment = experiment
        self._terms = terms
        #: Everything `_load` read, off the UI thread.
        self._all: list = []
        self._periods: dict[str, int] = {}
        self._spans: dict[str, tuple[int, int]] = {}
        self._names: dict[str, str] = {}
        self._metadata: dict = {}
        self._columns: list[str] = []
        self._embed_run_ids: dict[str, str] = {}
        self._answers: dict[str, str] = {}
        self._slots: dict[str, int] = {}
        #: The filtered bouts, all of them; the list holds the first `ROW_CAP`.
        self._shown: list = []
        self._rows: list = []
        #: ``{instance_id: (ethogram row, band index)}``, and its inverse per row.
        self._band_of: dict[str, tuple[int, int]] = {}
        self._band_rows: list[list] = []
        #: Filter widgets fire on every programmatic check; a rebuild must not refilter per item.
        self._building = False
        #: Whether the next clip put on the player starts playing — false only for the first row
        #: opened on its own (`_open_first`), so a click still plays as it always has.
        self._autoplay_next = True
        #: The instance on the player, so a refilter that keeps it does not replace it.
        self._playing: str | None = None
        #: Whether the first read has landed — what a caller waits on, since an empty run's
        #: screen looks the same before the read as after it.
        self._loaded = False
        #: INTERPRET's selected cluster: ``(name, colour, {recording_id: [run, ...]})`` or None.
        self._lane: tuple | None = None
        #: The newest lane INTERPRET asked for, whether it still has to be read (BROWSE was
        #: hidden), and a count that drops a read a newer request superseded.
        self._lane_request = None
        self._lane_wanted = False
        self._lane_generation = 0
        self._lane_rows: list[list[dict]] = []
        self._drawn: list[str] = []
        self._drawn_spans: dict[str, tuple[int, int]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Two modes, not one screen with everything empty in it — the reason AUDIT's intro
        # exists. Before anything is labelled there is nothing to filter, draw, list or play,
        # and four empty viewers describing their own absence read as a page that failed.
        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_empty())
        self._pages.addWidget(self._build_content(_ClipPlayer))
        layout.addWidget(self._pages)
        self.reload()

    # -- construction --------------------------------------------------------- #
    def _build_empty(self) -> QWidget:
        page = QWidget()
        column = QVBoxLayout(page)
        headline = QLabel("Nothing labelled yet")
        headline.setFont(emphasis_font(headline))
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(headline)
        self._empty_body = QLabel(
            "Label a node in INTERPRET and its instances appear here, to filter, "
            "watch and export."
        )
        self._empty_body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_body.setWordWrap(True)
        column.addWidget(self._empty_body)
        column.addStretch(1)
        return page

    def _build_content(self, player_class) -> QWidget:
        """The summary, then the filter column beside the results, both ending on one line.

        The block takes its natural height and the page's surplus goes **below** it, so the
        filter column — whose lists stretch — ends exactly where the results do rather than
        running on to the bottom of a tab as tall as INTERPRET.
        """
        page = QWidget()
        column = QVBoxLayout(page)
        top = QHBoxLayout()
        self._summary = QLabel("")
        self._summary.setFont(emphasis_font(self._summary))
        self._summary.setWordWrap(True)
        top.addWidget(self._summary, 1)
        self._clear = QPushButton("Clear filters")
        self._clear.clicked.connect(self._clear_filters)
        top.addWidget(self._clear)
        column.addLayout(top)

        # The filter rail folds by its grip to a strip reading "Filters"; the results do not.
        split = GripSplitter(Qt.Orientation.Horizontal, key="browse_rail")
        split.addWidget(self._build_rail())
        split.addWidget(self._build_main(player_class))
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 4)
        split.set_foldable(0, "Filters")
        if not split.remember():
            split.setSizes([280, 1000])
        column.addWidget(split)
        column.addStretch(1)
        return page

    def _build_rail(self) -> QWidget:
        """Labels, recordings, metadata. The lists share whatever height the results take."""
        rail = QWidget()
        column = QVBoxLayout(rail)
        column.setContentsMargins(0, 0, 0, 0)

        column.addWidget(self._heading("Labels"))
        self._labels = QTreeWidget()
        self._labels.setHeaderLabels(["label", "bouts", "time"])
        self._labels.setRootIsDecorated(True)
        # The name takes the slack; the two numbers hug their contents. Left to the default, the
        # last column stretched and the names elided to "re…", with a sub-label's text gone.
        header = self._labels.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._labels.itemChanged.connect(self._on_filter)
        column.addWidget(self._shares_height(self._labels), 3)

        column.addWidget(self._heading("Recordings"))
        self._search = QLineEdit()
        # Named, so a control walk can tell the one field here from every other line edit.
        self._search.setObjectName("browse_recording_search")
        self._search.setPlaceholderText("Search recordings")
        self._search.textChanged.connect(self._on_search)
        column.addWidget(self._search)
        self._recordings = QListWidget()
        self._recordings.itemChanged.connect(self._on_filter)
        column.addWidget(self._shares_height(self._recordings), 2)

        column.addWidget(self._heading("Metadata"))
        self._column = QComboBox()
        self._column.currentIndexChanged.connect(self._on_column)
        column.addWidget(self._column)
        self._values = QListWidget()
        self._values.itemChanged.connect(self._on_filter)
        column.addWidget(self._shares_height(self._values), 1)
        self._no_metadata = QLabel("This experiment has no metadata columns.")
        self._no_metadata.setWordWrap(True)
        set_label_color(self._no_metadata, MUTED)
        column.addWidget(self._no_metadata)
        # Stretch 0: it takes height only once every list is as tall as its rows (`_hug`), so a
        # short list ends at its last row and the rail's surplus collects below the lists.
        column.addStretch(0)
        return rail

    @staticmethod
    def _shares_height(widget):
        """A rail list that asks for no height of its own and takes its share of the results'.

        A list's own hint is ~192px whatever it holds, so three of them made the rail taller than
        the results and the two columns ended at different heights. Ignoring the hint leaves the
        results to decide the block's height and the lists to divide it, down to a few lines —
        and no further than its rows (:meth:`_hug`).
        """
        widget.setSizePolicy(
            widget.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Ignored
        )
        widget.setMinimumHeight(lines(4))
        return widget

    @staticmethod
    def rows_px(widget) -> int:
        """The height that shows every row ``widget`` has on show, and nothing under them."""
        if isinstance(widget, QTreeWidget):
            shown, walk = 0, QTreeWidgetItemIterator(widget)
            while walk.value() is not None:
                item = walk.value()
                parent = item.parent()
                if not item.isHidden() and (parent is None or parent.isExpanded()):
                    shown += 1
                walk += 1
            header = (
                0 if widget.isHeaderHidden() else widget.header().sizeHint().height()
            )
        else:
            shown = sum(
                1 for i in range(widget.count()) if not widget.item(i).isHidden()
            )
            header = 0
        row = widget.sizeHintForRow(0) if shown else 0
        return header + shown * max(row, 1) + 2 * widget.frameWidth()

    def _hug(self) -> None:
        """Cap each rail list at its rows. Measured at 1440x847: Labels was 290px for four rows
        and Recordings 210px for two — two thirds of the rail blank grid inside its boxes.
        """
        for widget in (self._labels, self._recordings, self._values):
            widget.setMaximumHeight(max(self.rows_px(widget), widget.minimumHeight()))

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setFont(emphasis_font(label))
        return label

    def _build_main(self, player_class) -> QWidget:
        """The ethogram over the list and the player — or, when the filters leave nothing,
        one sentence and the way back, in place of a picture, a list and a player of nothing.
        """
        self._results = QStackedWidget()
        data = QWidget()
        column = QVBoxLayout(data)
        column.setContentsMargins(0, 0, 0, 0)

        self._ethogram = EthogramStrip()
        self._ethogram.band_activated.connect(self._on_band)
        self._ethogram.lane_activated.connect(self._on_lane)
        self._ethogram.set_describer(self._describe_band, self._describe_lane)
        self._ethogram_scroll = QScrollArea()
        self._ethogram_scroll.setWidgetResizable(True)
        self._ethogram_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._ethogram_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._ethogram_scroll.setWidget(self._ethogram)
        self._ethogram_band = CollapsibleBand(
            "Ethogram — labelled time, click a bout to play it",
            ruled(self._ethogram, self._ethogram_scroll),
            key="label_browse/ethogram",
            on_resize=self._resize_ethogram,
        )
        self._ethogram_height = _ETHOGRAM_ROWS * self._ethogram.row_px()
        # The strip changes its own height — a lane added under every row grows each row — and
        # the band's height is read from it, so it is re-fitted on that, not only on a redraw.
        self._ethogram.installEventFilter(self)
        column.addWidget(self._ethogram_band)

        self._model = QStandardItemModel(0, 0)
        self._table = QTableView()
        self._table.setModel(self._model)
        make_columns_resizable(self._table)
        enable_sorting(self._table)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.selectionModel().currentRowChanged.connect(self._show_row)
        self._table.selectionModel().selectionChanged.connect(self._sync_export)
        fit_table_height(self._table, max_rows=BROWSE_ROWS, min_rows=0)

        self._player = player_class(
            self._experiment, autoplay=True, context_s=0, top=True
        )
        # The list folds by its grip to a strip reading "Instances"; the player does not.
        split = GripSplitter(Qt.Orientation.Horizontal, key="browse_instances")
        split.addWidget(top_of_pane(self._table))
        split.addWidget(self._player)
        split.set_foldable(0, "Instances")
        #: The list beside the player: fitted to its columns (`_fit_list_split`) until dragged.
        self._list_split = split
        self._list_split_dragged = split.remember()
        if not self._list_split_dragged:
            split.setSizes([520, 460])
        column.addWidget(split, 1)
        split.splitterMoved.connect(self._on_list_split_moved)
        split.fold_changed.connect(self._on_list_split_moved)
        split.installEventFilter(self)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self._export_selected = QPushButton("Export selected")
        self._export_selected.clicked.connect(lambda: self._export(selected=True))
        buttons.addWidget(self._export_selected)
        self._export_all = QPushButton("Export all")
        self._export_all.clicked.connect(lambda: self._export(selected=False))
        # Nothing is shown until the load lands, and a snapshot's stay off after it does.
        set_available(self._export_selected, False, "Select rows in the list first.")
        set_available(self._export_all, False, "Nothing is labelled yet.")
        buttons.addWidget(self._export_all)
        column.addLayout(buttons)
        self._results.addWidget(data)

        none = QWidget()
        box = QVBoxLayout(none)
        message = QLabel("No instances match these filters.")
        message.setFont(emphasis_font(message))
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(message)
        again = QPushButton("Clear filters")
        again.clicked.connect(self._clear_filters)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(again)
        row.addStretch()
        box.addLayout(row)
        box.addStretch(1)
        self._results.addWidget(none)
        return self._results

    def _resize_ethogram(self, delta: int) -> None:
        self._ethogram_height = max(
            _ETHOGRAM_ROWS * self._ethogram.row_px(),
            self._ethogram_height + int(delta),
        )
        self._fit_ethogram()

    def _fit_ethogram(self) -> None:
        """Every strip when they fit; past `_ETHOGRAM_ROWS`, a scrolling window with a handle.

        The handle appears only when dragging it would change something. With every strip
        already showing it moved nothing, and a drag bar that moves nothing reads as broken.

        Read from the strip's height **now**, and re-run whenever that changes (`eventFilter`):
        sized once at the redraw, the band kept the height of rows without INTERPRET's lane
        after the lane arrived — 42px of a 56px strip at 1440x847, the second recording half
        under the fold behind a scrollbar nobody expects on two rows.
        """
        overflow = len(self._band_rows) > _ETHOGRAM_ROWS
        content = self._ethogram.height() + 2
        window = max(self._ethogram_height, _ETHOGRAM_ROWS * self._ethogram.row_px())
        self._ethogram_band.set_resizable(overflow)
        self._ethogram_scroll.setFixedHeight(
            min(content, window) if overflow else content
        )

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 — Qt's name
        if event.type() == QEvent.Type.Resize:
            if watched is self._ethogram:
                self._fit_ethogram()
            elif watched is self._list_split:
                self._fit_list_split()
        return super().eventFilter(watched, event)

    def _on_list_split_moved(self, *_args) -> None:
        self._list_split_dragged = True

    def _fit_list_split(self) -> None:
        """Give the list its columns' width, up to leaving the player `_PLAYER_MIN_LINES`; the
        name column takes what is left over or gives up what is short (`give_slack_to`), so the
        last column is never cut and no horizontal scrollbar appears. Until the divider is
        dragged: from then on the split is the user's.

        A fixed 520/460 split cut the audit column at 1440x847 to "aud", its values to "not
        climl", behind a 1px horizontal scroll.
        """
        if self._list_split_dragged or self._list_split.is_folded(0):
            return
        split, table = self._list_split, self._table
        total = sum(split.sizes())
        if total <= 0:
            return
        header = table.horizontalHeader()
        columns = sum(
            max(header.sectionSizeHint(c), table.sizeHintForColumn(c))
            for c in range(header.count())
        )
        need = (
            columns
            + 2 * table.frameWidth()
            + table.verticalScrollBar().sizeHint().width()
        )
        width = max(min(need, total - lines(_PLAYER_MIN_LINES)), lines(12))
        if split.sizes()[0] != width:
            split.setSizes([width, total - width])

    # -- reading (off the UI thread) ------------------------------------------ #
    def reload(self) -> None:
        run_busy(
            self,
            self._load,
            self._show,
            label="Reading the labelled data…",
            supersedes="labels",
        )

    def _load(self) -> dict:
        """The artifact, the names, the metadata and the audit, in one off-thread read."""
        try:
            artifact = self._run.artifact()  # off-thread: `run_busy`
        except FileNotFoundError:
            artifact = None
        state = self._run.state()
        decisions = {  # off-thread: `run_busy`
            row.instance_id: row
            for row in self._run.audit_decisions()  # off-thread: `run_busy`
            if row.decided
        }
        rows = labelled_bouts(artifact.instances) if artifact is not None else []
        return {
            "rows": rows,
            "coverage": list(artifact.coverage) if artifact is not None else [],
            "names": self._experiment.recording_names(),
            "metadata": self._experiment.recording_metadata,
            "columns": self._experiment.metadata_columns,
            "embed": run_selection.runs_by_recording(state, state.source_recording_ids),
            "answers": {
                row.instance_id: audit.answer_text(
                    decisions[row.instance_id], path_of(row)
                )
                for row in rows
                if row.instance_id in decisions
            },
        }

    def _show(self, payload: dict) -> None:
        self._all = payload["rows"]
        self._periods = bo.period_by_recording(payload["coverage"])
        self._spans = recording_spans(payload["coverage"])
        self._names = payload["names"]
        self._metadata = payload["metadata"]
        self._columns = payload["columns"]
        self._embed_run_ids = payload["embed"]
        self._answers = payload["answers"]
        self._slots = root_slots(path_of(row) for row in self._all)
        self._build_filters()
        self._refilter()
        self._loaded = True

    # -- filters -------------------------------------------------------------- #
    def _build_filters(self) -> None:
        """Rebuild the three filters, keeping whatever was unticked before a reload."""
        dropped_paths = self._unticked_paths()
        dropped_recordings = self._unticked(self._recordings)
        column = self._column.currentData()
        dropped_values = self._unticked(self._values)
        self._building = True
        try:
            self._labels.clear()
            parent = None
            for text, depth, root, bouts, time_ns, path in label_table(
                self._all, self._periods
            ):
                item = QTreeWidgetItem(
                    [text, f"{bouts:,}", format_duration(time_ns / 1e9)]
                )
                item.setData(0, Qt.ItemDataRole.UserRole, path)
                item.setData(
                    0,
                    Qt.ItemDataRole.DecorationRole,
                    plot_theme.label_identity(self._slots.get(root, 0)),
                )
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                if depth == 0:
                    self._labels.addTopLevelItem(item)
                    parent = item
                else:
                    parent.addChild(item)
                    parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsAutoTristate)
                item.setCheckState(
                    0,
                    (
                        Qt.CheckState.Unchecked
                        if path in dropped_paths
                        else Qt.CheckState.Checked
                    ),
                )
            self._labels.expandAll()

            self._recordings.clear()
            present = sorted(
                {row.recording_id for row in self._all} | set(self._spans),
                key=lambda r: self._name(r).lower(),
            )
            for recording_id in present:
                item = QListWidgetItem(self._name(recording_id))
                item.setData(Qt.ItemDataRole.UserRole, recording_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Unchecked
                    if recording_id in dropped_recordings
                    else Qt.CheckState.Checked
                )
                self._recordings.addItem(item)
            self._on_search(self._search.text())

            self._column.clear()
            self._column.addItem("Any group", None)
            for name in self._columns:
                self._column.addItem(name, name)
            index = self._column.findData(column)
            self._column.setCurrentIndex(max(index, 0))
            self._column.setVisible(bool(self._columns))
            self._no_metadata.setVisible(not self._columns)
            self._fill_values(dropped_values)
        finally:
            self._building = False
        self._hug()

    def _fill_values(self, dropped: set[str] = frozenset()) -> None:
        """The chosen column's values among this run's recordings, each with its count."""
        column = self._column.currentData()
        self._values.clear()
        self._values.setVisible(column is not None)
        if column is None:
            return
        counts: dict[str, int] = {}
        for recording_id in {row.recording_id for row in self._all} | set(self._spans):
            value = (self._metadata.get(recording_id) or {}).get(column) or NO_VALUE
            counts[value] = counts.get(value, 0) + 1
        for value in sorted(counts):
            item = QListWidgetItem(f"{value} ({counts[value]})")
            item.setData(Qt.ItemDataRole.UserRole, value)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Unchecked if value in dropped else Qt.CheckState.Checked
            )
            self._values.addItem(item)

    def _items(self):
        """Every label item, parents and sub-labels alike."""
        out = []
        for i in range(self._labels.topLevelItemCount()):
            top = self._labels.topLevelItem(i)
            out.append(top)
            out.extend(top.child(j) for j in range(top.childCount()))
        return out

    def _label_items(self):
        """The items that stand for a set of bouts: sub-labels, and parents without any."""
        return [item for item in self._items() if item.childCount() == 0]

    def _unticked_paths(self) -> set[str]:
        return {
            item.data(0, Qt.ItemDataRole.UserRole)
            for item in self._label_items()
            if item.checkState(0) != Qt.CheckState.Checked
        }

    @staticmethod
    def _unticked(widget: QListWidget) -> set[str]:
        return {
            widget.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(widget.count())
            if widget.item(i).checkState() != Qt.CheckState.Checked
        }

    def _filters(self) -> dict:
        """The current filters as `filter_bouts` takes them — ``None`` where nothing is unticked."""
        paths = {
            item.data(0, Qt.ItemDataRole.UserRole)
            for item in self._label_items()
            if item.checkState(0) == Qt.CheckState.Checked
        }
        recordings = {
            self._recordings.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._recordings.count())
            if self._recordings.item(i).checkState() == Qt.CheckState.Checked
        }
        column = self._column.currentData()
        values = {
            self._values.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._values.count())
            if self._values.item(i).checkState() == Qt.CheckState.Checked
        }
        return {
            "paths": None if not self._unticked_paths() else paths,
            "recordings": None if not self._unticked(self._recordings) else recordings,
            "meta_column": column,
            "meta_values": (
                None if column is None or not self._unticked(self._values) else values
            ),
            "metadata": self._metadata,
        }

    def _filter_words(self, filters: dict) -> list[str]:
        """The active filters, as the summary line says them."""
        words = []
        if filters["paths"] is not None:
            words.append(
                ", ".join(_display(p) for p in sorted(filters["paths"])) or "no labels"
            )
        if filters["recordings"] is not None:
            words.append(f"{len(filters['recordings'])} recording(s)")
        if filters["meta_values"] is not None:
            values = ", ".join(sorted(filters["meta_values"])) or "nothing"
            words.append(f"{filters['meta_column']} = {values}")
        return words

    def _on_filter(self, *_args) -> None:
        if not self._building:
            self._refilter()

    def _on_search(self, text: str) -> None:
        needle = text.strip().lower()
        for i in range(self._recordings.count()):
            item = self._recordings.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())
        self._hug()

    def _on_column(self, _index: int) -> None:
        if self._building:
            return
        self._building = True
        try:
            self._fill_values()
        finally:
            self._building = False
        self._hug()
        self._refilter()

    def _clear_filters(self) -> None:
        self._building = True
        try:
            for item in self._items():
                item.setCheckState(0, Qt.CheckState.Checked)
            for widget in (self._recordings, self._values):
                for i in range(widget.count()):
                    widget.item(i).setCheckState(Qt.CheckState.Checked)
            self._search.clear()
        finally:
            self._building = False
        self._refilter()

    # -- what the filters give ------------------------------------------------ #
    def _refilter(self) -> None:
        """Apply the filters, then redraw the summary, the ethogram and the list together."""
        filters = self._filters()
        self._shown = filter_bouts(self._all, **filters)
        self._rows = self._shown[:ROW_CAP]
        set_available(
            self._clear,
            bool(self._filter_words(filters)),
            "Every label and recording is shown already.",
        )
        self._summarise(filters)
        self._draw_ethogram(filters)
        self._fill_table()
        self._export_all.setText(f"Export all {len(self._shown):,}")
        set_available(
            self._export_all,
            bool(self._shown) and self._run.editable,
            (
                "This is a snapshot — export from the run itself."
                if not self._run.editable
                else "Nothing is labelled, or the filters leave nothing."
            ),
        )
        self._sync_export()
        self._pages.setCurrentIndex(1 if self._all else 0)
        self._results.setCurrentIndex(0 if self._shown else 1)
        # The no-match page carries its own, under the sentence that explains it.
        self._clear.setVisible(bool(self._shown))
        self._open_first()

    def _open_first(self) -> None:
        """Put the first instance on the player, paused, when what it showed has gone.

        A player saying "select an instance" beside a full list is a sentence about the widget,
        not the data; the first instance, still, says what the list is of. Paused, because a
        filter change is not a request to watch something. **Shown, not selected**: a selected
        row is an export scope, and opening the tab must not quietly narrow Export to one
        instance nobody picked.
        """
        if not self._rows:
            return
        if self._playing in {row.instance_id for row in self._shown}:
            return
        self._autoplay_next = False
        self._play(self._rows[0])

    def _summarise(self, filters: dict) -> None:
        unit = self._terms.member if self._terms.member != "rows" else "instances"
        if not self._all:
            self._summary.setText("")
            return
        total_ns = sum(bo.instance_ns(r, self._periods) for r in self._all)
        shown_ns = sum(bo.instance_ns(r, self._periods) for r in self._shown)
        words = self._filter_words(filters)
        text = (
            f"{len(self._shown):,} of {len(self._all):,} labelled {unit} · "
            f"{format_duration(shown_ns / 1e9)} of {format_duration(total_ns / 1e9)}"
        )
        if words:
            text += " · filters: " + "; ".join(words)
        if len(self._shown) > len(self._rows):
            text += f" · the list shows the first {len(self._rows):,}; the ethogram shows them all"
        self._summary.setText(text)

    def _draw_ethogram(self, filters: dict) -> None:
        """One strip per recording that passes the recording and group filters."""
        recordings = [
            recording_id
            for recording_id in sorted(
                {row.recording_id for row in self._all} | set(self._spans),
                key=lambda r: self._name(r).lower(),
            )
            if recording_passes(
                recording_id,
                recordings=filters["recordings"],
                meta_column=filters["meta_column"],
                meta_values=filters["meta_values"],
                metadata=self._metadata,
            )
        ]
        bands: dict[str, list] = {r: [] for r in recordings}
        owners: dict[str, list] = {r: [] for r in recordings}
        # One span per recording. A recording with no span on record falls back to the
        # instances it holds, taken whole — a span per *instance* drew each bout across the
        # entire row, and has no length to put on the shared time axis.
        spans = dict(self._spans)
        for row in self._shown:
            if row.recording_id in bands and row.recording_id not in self._spans:
                end = row.t_ns + bo.instance_ns(row, self._periods)
                lo, hi = spans.get(row.recording_id, (row.t_ns, end))
                spans[row.recording_id] = (min(lo, row.t_ns), max(hi, end))
        for row in self._shown:
            if row.recording_id not in bands:
                continue
            first, last = spans[row.recording_id]
            length = max(last - first, 1)
            end = row.t_ns + bo.instance_ns(row, self._periods)
            root = path_of(row).split(bo.PATH_SEPARATOR)[0]
            bands[row.recording_id].append(
                (
                    (row.t_ns - first) / length,
                    (end - first) / length,
                    self._slots.get(root, 0),
                )
            )
            owners[row.recording_id].append(row)
        self._band_rows = [owners[r] for r in recordings]
        self._band_of = {
            row.instance_id: (i, j)
            for i, rows in enumerate(self._band_rows)
            for j, row in enumerate(rows)
        }
        self._ethogram.set_rows(
            [(self._name(r), bands[r]) for r in recordings],
            cluster_colors(max(len(self._slots), 1)),
            None,
            lengths_ns=[
                spans[r][1] - spans[r][0] if r in spans else 1 for r in recordings
            ],
        )
        self._drawn, self._drawn_spans = recordings, spans
        self._draw_lane()
        self._fit_ethogram()

    # -- INTERPRET's selected cluster ------------------------------------------ #
    def set_cluster_lane(self, selected) -> None:
        """``(view, node, name, colour)`` from INTERPRET, or ``None``: the cluster's time as a
        lane under each strip, labelled or not. A single named node through ``labelled_runs``
        is every stretch under that node, so the lane reads nothing the node cache has not.
        The clusters as fitted (``apply_edits=False``): a clip taken out of a label is still in
        its cluster, and INTERPRET's lane says what the cluster holds.

        ``labelled_runs`` walks every leaf run, and INTERPRET calls this on every node click
        while BROWSE is usually a hidden tab: the request is remembered and read only once
        BROWSE shows (:meth:`showEvent`), on a worker, the newest request winning.
        """
        self._lane_request = selected
        self._lane_generation += 1
        if selected is None:
            self._lane_wanted = False
            self._lane = None
            if hasattr(self, "_ethogram"):
                self._draw_lane()
            return
        self._lane_wanted = True
        if self.isVisible():
            self._read_lane()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt virtual
        super().showEvent(event)
        if self._lane_wanted:
            self._read_lane()

    def _read_lane(self) -> None:
        """Read the remembered lane request off the UI thread; a newer request drops it."""
        self._lane_wanted = False
        view, node, name, colour = self._lane_request
        generation = self._lane_generation
        worker = BackgroundCall(
            lambda: _lane_runs(view, node, name), receiver=self
        )  # off-thread: `labelled_runs`

        def done(runs) -> None:
            if generation != self._lane_generation:
                return
            self._lane = (name, colour, runs)
            if hasattr(self, "_ethogram"):
                self._draw_lane()

        worker.done.connect(done)
        worker.failed.connect(lambda exc: log.warning("cluster lane not read: %s", exc))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _draw_lane(self) -> None:
        if self._lane is None:
            self._lane_rows = []
            self._ethogram.set_lane(None)
            return
        _name, colour, runs = self._lane
        marks, rows = [], []
        for recording_id in self._drawn:
            first, last = self._drawn_spans.get(recording_id, (0, 1))
            length = max(last - first, 1)
            mine = sorted(runs.get(recording_id, []), key=lambda r: r["start_ns"])
            marks.append(
                [
                    ((r["start_ns"] - first) / length, (r["end_ns"] - first) / length)
                    for r in mine
                ]
            )
            rows.append(mine)
        self._lane_rows = rows
        self._ethogram.set_lane(marks, colour)

    def _lane_run(self, row: int, index: int) -> dict | None:
        if 0 <= row < len(self._lane_rows) and 0 <= index < len(self._lane_rows[row]):
            return self._lane_rows[row][index]
        return None

    def _on_lane(self, row: int, index: int) -> None:
        run = self._lane_run(row, index)
        if run is None:
            return
        self._player.set_clip(
            ClipSpec(
                recording_id=run["recording_id"],
                track_id=int(run["subject"] or 0),
                embed_run_id=self._embed_run_ids.get(run["recording_id"], ""),
                start_t_ns=int(run["start_ns"]),
                end_t_ns=int(run["end_ns"]),
                label=run["label"],
            ),
            autoplay=True,
        )

    def _describe_lane(self, row: int, index: int) -> str:
        run = self._lane_run(row, index)
        if run is None:
            return ""
        seconds = (run["end_ns"] - run["start_ns"]) / 1e9
        return (
            f"{run['label']} · {self._name(run['recording_id'])} at "
            f"{_clock(run['start_ns'])} · {seconds:.1f} s — click to play"
        )

    def _fill_table(self) -> None:
        subjects = {row.subject for row in self._all}
        self._with_subject = len(subjects) > 1
        headers = [self._terms.unit, "recording"]
        if self._with_subject:
            headers.append("subject")
        headers += ["start", "duration", "audit"]
        self._model.clear()
        self._model.setHorizontalHeaderLabels(headers)
        for index, row in enumerate(self._rows):
            path = path_of(row)
            seconds = bo.instance_ns(row, self._periods) / 1e9
            label = SortCell(_display(path), path)
            label.setData(
                plot_theme.label_identity(
                    self._slots.get(path.split(bo.PATH_SEPARATOR)[0], 0)
                ),
                Qt.ItemDataRole.DecorationRole,
            )
            # The row's identity travels with it: the view re-sorts the model, so a row number
            # stops meaning what it meant.
            label.setData(index, Qt.ItemDataRole.UserRole)
            # The name column gives up width when the pane is short of it (`_fit_list_split`),
            # so its whole text is on hover.
            label.setToolTip(_display(path))
            cells = [
                label,
                SortCell(self._name(row.recording_id), self._name(row.recording_id)),
            ]
            if self._with_subject:
                cells.append(num_cell(row.subject))
            answer = self._answers.get(row.instance_id, "")
            cells += [
                SortCell(_clock(row.t_ns), int(row.t_ns)),
                num_cell(seconds, f"{seconds:.2f} s"),
                SortCell(answer or "—", answer),
            ]
            self._model.appendRow(cells)
        self._table.resizeColumnsToContents()
        give_slack_to(self._table, 0)
        fit_table_height(self._table, max_rows=BROWSE_ROWS, min_rows=0)
        self._fit_list_split()

    def _name(self, recording_id: str) -> str:
        """The recording's name, or its id if it has none left (deleted since the run)."""
        return self._names.get(recording_id, recording_id)

    # -- watching ------------------------------------------------------------- #
    def _row_at(self, view_row: int):
        item = self._model.item(view_row, 0)
        index = None if item is None else item.data(Qt.ItemDataRole.UserRole)
        if index is None or not (0 <= index < len(self._rows)):
            return None
        return self._rows[index]

    def _show_row(self, current, _previous) -> None:
        """Play the selected instance and outline its band. Single click, like the query Browse."""
        row = self._row_at(current.row())
        if row is not None:
            self._play(row)

    def _play(self, row) -> None:
        self._playing = row.instance_id
        self._ethogram.set_highlight(self._band_of.get(row.instance_id))
        autoplay, self._autoplay_next = self._autoplay_next, True
        self._player.set_clip(
            ClipSpec(
                recording_id=row.recording_id,
                track_id=int(row.subject or 0),
                embed_run_id=self._embed_run_ids.get(row.recording_id, ""),
                start_t_ns=int(row.t_ns),
                end_t_ns=int(row.end_t_ns),
                label=_display(path_of(row)),
                score=row.score,
            ),
            autoplay=autoplay,
        )

    def _on_band(self, band_row: int, index: int) -> None:
        """A band clicked: select its row in the list, or play it directly past the cap."""
        if not (0 <= band_row < len(self._band_rows)):
            return
        if not (0 <= index < len(self._band_rows[band_row])):
            return
        target = self._band_rows[band_row][index]
        for view_row in range(self._model.rowCount()):
            row = self._row_at(view_row)
            if row is not None and row.instance_id == target.instance_id:
                self._table.selectRow(view_row)
                self._table.scrollTo(self._model.index(view_row, 0))
                return
        self._play(target)

    def _describe_band(self, band_row: int, index: int) -> str:
        if not (0 <= band_row < len(self._band_rows)):
            return ""
        rows = self._band_rows[band_row]
        if not (0 <= index < len(rows)):
            return ""
        row = rows[index]
        seconds = bo.instance_ns(row, self._periods) / 1e9
        return (
            f"{_display(path_of(row))} · {self._name(row.recording_id)} at "
            f"{_clock(row.t_ns)} · {seconds:.1f} s — click to play"
        )

    # -- export --------------------------------------------------------------- #
    def _sync_export(self, *_args) -> None:
        picked = len(self._selected_ids())
        self._export_selected.setText(
            f"Export {picked:,} selected" if picked else "Export selected"
        )
        set_available(
            self._export_selected,
            bool(picked) and self._run.editable,
            (
                "This is a snapshot — export from the run itself."
                if not self._run.editable
                else "Select rows in the list first."
            ),
        )

    def _selected_ids(self) -> tuple[str, ...]:
        picked = []
        for index in self._table.selectionModel().selectedRows():
            row = self._row_at(index.row())
            if row is not None:
                picked.append(row.instance_id)
        return tuple(picked)

    def export_scope(self):
        """The instances picked in the list, or ``None`` for everything."""
        from detail.api.exporters import ExportScope

        picked = self._selected_ids()
        return ExportScope(instance_ids=picked) if picked else None

    def _export(self, *, selected: bool) -> None:
        """Open the one export dialog on the selection, or on every filtered instance —
        not only the rows the list had room for."""
        from detail.api.exporters import ExportScope, ExportSubject
        from detail.gui.export import open_export

        ids = (
            self._selected_ids()
            if selected
            else tuple(row.instance_id for row in self._shown)
        )
        if not ids:
            return
        open_export(
            self,
            ExportSubject("analysis_run", self._experiment.dir, self._run.id),
            ExportScope(instance_ids=ids),
        )
