"""Look-alike clusters — the window that suggests one label for clusters that are one behaviour
(§22 "Look-alike clusters").

Opened from a clustering run's Interpret tab (*Look-alike clusters…*). It lists the run's groups
(``api/cluster_lookalikes.py``): clusters in different branches whose body measures could not be
told apart, pair by pair, with position and facing left out, chained. For each group:

- **the members**, largest first, each with its share of the run, its label, what else its
  label would cover, and a tick: untick one to leave it out of the label. A label
  covers everything under its cluster with no label of its own, and clusters nest, so a member
  that holds clusters outside the group starts unticked and names them (``sweeps``).
  Clicking a member selects it in the tree, so its card and clips show beside this window.
- **why they are grouped**: every link, with how well body measures and position and facing alone
  told the pair apart — the evidence, not only the verdict.
- **the name**, filled in with the group's proposal (a label a member already has), and
  editable.

*Label as one…* plans the label (``plan_join``), shows what each cluster would do — a new label,
which label it joins, or why it is skipped — and only then asks the workspace to write it, through
the run's own label edit. Nothing here writes a label itself.

*Find look-alikes…* asks the workspace to open the launch dialog on *Find look-alike clusters* for
this run; the groups are read again when it finishes, and on *Refresh*. Every read is off the UI
thread.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
)

from detail.gui.dialog import Dialog
from detail.gui.sizing import set_available
from detail.gui.status_color import MUTED, WARN, set_label_color
from detail.gui.workers import BackgroundCall

#: Item data roles: a group row holds its index, a member row its node.
_GROUP_ROLE = Qt.ItemDataRole.UserRole
_NODE_ROLE = Qt.ItemDataRole.UserRole + 1
_COLUMNS = (
    "Cluster",
    "Share of run",
    "Would also label",
    "Your label",
)
#: The column saying what a member's label would sweep in.
_SWEEP_COLUMN = _COLUMNS.index("Would also label")


class LookalikesWindow(Dialog):
    """The run's look-alike groups, with one label for the clusters of one."""

    #: Select this node in the tree (its card and clips show there).
    node_wanted = pyqtSignal(int)
    #: Label these nodes as one, with this name — planned and confirmed already.
    label_wanted = pyqtSignal(list, str)
    #: Open the launch dialog on *Find look-alike clusters* for this run.
    find_wanted = pyqtSignal()

    def __init__(
        self,
        run,
        *,
        unit: str = "cluster",
        names_now=None,
        parent_array=None,
        editable: bool = True,
        why_not: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle(f"Look-alike {unit}s · {run.id}")
        self._run = run
        self._unit = unit
        self._names_now = names_now or (lambda: {})
        self._parent_array = parent_array or (lambda: None)
        self._editable = editable
        self._why_not = why_not
        self._groups: list = []
        #: The group whose proposal the name box was last filled with.
        self._named_group: int | None = None
        self._reader: BackgroundCall | None = None
        body = self.body()

        intro = QLabel(
            f"{unit.capitalize()}s in different branches that the body measures cannot tell "
            "apart, with where the animal is and which way it faces left out: often one "
            "behaviour the tree split by place or direction. Pairs that look alike chain "
            "into groups. A group is a suggestion until you label it."
        )
        intro.setWordWrap(True)
        body.addWidget(intro)

        actions = QHBoxLayout()
        self._status = QLabel("Reading the comparisons…")
        self._status.setWordWrap(True)
        actions.addWidget(self._status, 1)
        self._refresh = QPushButton("Refresh")
        self._refresh.setToolTip("Read the saved comparisons again.")
        self._refresh.clicked.connect(self.reload)
        actions.addWidget(self._refresh)
        self._find = QPushButton("Find look-alikes…")
        self._find.setToolTip(
            f"Compare pairs of this run's {unit}s as one job you can leave running "
            "(Find look-alike clusters). Pairs already compared are skipped."
        )
        self._find.clicked.connect(lambda: self.find_wanted.emit())
        actions.addWidget(self._find)
        body.addLayout(actions)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(len(_COLUMNS))
        self._tree.setHeaderLabels(list(_COLUMNS))
        self._tree.setMinimumHeight(220)
        self._tree.currentItemChanged.connect(lambda *_a: self._show_selected())
        self._tree.itemClicked.connect(self._on_clicked)
        self._tree.itemChanged.connect(lambda *_a: self._sync_label_button())
        body.addWidget(self._tree)

        self._why = QLabel("")
        self._why.setWordWrap(True)
        self._why.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.addWidget(self._why)

        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label:"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("a name for the whole group")
        self._name.textChanged.connect(lambda _t: self._sync_label_button())
        label_row.addWidget(self._name, 1)
        self._label = QPushButton("Label as one…")
        self._label.clicked.connect(self._plan_label)
        label_row.addWidget(self._label)
        body.addLayout(label_row)
        self._proposed = QLabel("")
        self._proposed.setWordWrap(True)
        set_label_color(self._proposed, MUTED)
        body.addWidget(self._proposed)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)
        self._sync_label_button()
        self.reload()

    # -- reading ------------------------------------------------------------------------ #
    def reload(self) -> None:
        """Read the groups again, off the UI thread."""
        self._status.setText("Reading the comparisons…")
        self._refresh.setEnabled(False)
        run = self._run
        self._reader = BackgroundCall(
            lambda: run.lookalike_groups(), receiver=self
        )  # off-thread: one file per compared pair
        self._reader.done.connect(self._show_groups)
        self._reader.failed.connect(self._show_failure)
        self._reader.start()

    def _show_failure(self, exc) -> None:
        self._refresh.setEnabled(True)
        self._status.setText(f"Could not read the comparisons: {exc}")

    def _show_groups(self, groups: list) -> None:
        self._refresh.setEnabled(True)
        self._groups = list(groups)
        self._named_group = None
        self._proposed.setText("")
        self._tree.blockSignals(True)
        self._tree.clear()
        for index, group in enumerate(self._groups):
            proposed = f" — “{group.name}”" if group.name else ""
            top = QTreeWidgetItem(
                [f"{len(group.nodes)} {self._unit}s look alike{proposed}"]
            )
            top.setData(0, _GROUP_ROLE, index)
            top.setFirstColumnSpanned(True)
            self._tree.addTopLevelItem(top)
            for node in group.nodes:
                swept = group.sweeps.get(node, ())
                item = QTreeWidgetItem(
                    [
                        f"#{node}",
                        f"{group.share(node):.1%}",
                        _swept_summary(swept, self._unit),
                        group.labels.get(node, ""),
                    ]
                )
                if swept:
                    item.setToolTip(_SWEEP_COLUMN, "\n".join(s.text() for s in swept))
                item.setData(0, _GROUP_ROLE, index)
                item.setData(0, _NODE_ROLE, int(node))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    (
                        Qt.CheckState.Checked
                        if node in group.ticked
                        else Qt.CheckState.Unchecked
                    ),
                )
                top.addChild(item)
            top.setExpanded(True)
        self._tree.blockSignals(False)
        for column in range(len(_COLUMNS)):
            self._tree.resizeColumnToContents(column)
        if self._groups:
            links = sum(len(g.links) for g in self._groups)
            self._status.setText(
                f"{len(self._groups)} group{'s' if len(self._groups) != 1 else ''}, held "
                f"together by {links} look-alike pair{'s' if links != 1 else ''}."
            )
            self._tree.setCurrentItem(self._tree.topLevelItem(0))
        else:
            self._status.setText(
                f"No look-alike {self._unit}s yet. Find look-alikes… compares pairs of this "
                f"run's {self._unit}s; a pair compared in the Cluster profile window (Compare "
                "against ▸ Another cluster) counts too."
            )
            self._why.setText("")
            self._name.setText("")
        self._sync_label_button()

    # -- the selected group ------------------------------------------------------------- #
    def _current_group(self):
        item = self._tree.currentItem()
        if item is None:
            return None
        index = item.data(0, _GROUP_ROLE)
        return self._groups[index] if index is not None else None

    def _group_item(self, group_index: int) -> QTreeWidgetItem:
        return self._tree.topLevelItem(group_index)

    def _show_selected(self) -> None:
        group = self._current_group()
        if group is None:
            self._why.setText("")
            self._sync_label_button()
            return
        lines = ["<b>Why they are grouped:</b>"]
        lines += [f"• {link.text()}" for link in group.links]
        if group.sweeps:
            lines.append(
                f"<b>Unticked:</b> a label covers everything under its {self._unit} that "
                f"has no label of its own, so these would also label {self._unit}s outside "
                "the group. Label those first, or tick one anyway."
            )
            lines += [
                f"• #{node} would also label "
                + ", ".join(s.text() for s in group.sweeps[node])
                for node in group.nodes
                if node in group.sweeps
            ]
        self._why.setText("<br>".join(lines))
        index = self._groups.index(group)
        if self._named_group != index:
            self._named_group = index
            self._name.setText(group.name)
            self._proposed.setText(
                f"Proposed: {group.named}." if group.name else f"{group.named}."
            )
        self._sync_label_button()

    def _on_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        node = item.data(0, _NODE_ROLE)
        if node is not None:
            self.node_wanted.emit(int(node))

    def chosen_nodes(self) -> list[int]:
        """The ticked members of the selected group."""
        group = self._current_group()
        if group is None:
            return []
        top = self._group_item(self._groups.index(group))
        return [
            int(top.child(i).data(0, _NODE_ROLE))
            for i in range(top.childCount())
            if top.child(i).checkState(0) == Qt.CheckState.Checked
        ]

    def _sync_label_button(self) -> None:
        chosen = self.chosen_nodes()
        if not self._editable:
            reason = self._why_not or "This run's labels cannot change here."
        elif not chosen:
            reason = "Choose a group, with at least one of its clusters ticked."
        elif not self._name.text().strip():
            reason = "Give the label a name."
        else:
            reason = ""
        set_available(self._label, not reason, reason)

    # -- labelling ---------------------------------------------------------------------- #
    def _plan_label(self) -> None:
        from detail.api.cluster_lookalikes import plan_join

        nodes, name = self.chosen_nodes(), self._name.text().strip()
        if not nodes or not name:
            return
        plan = plan_join(self._names_now(), self._parent_array(), nodes, name)
        group = self._current_group()
        dialog = LabelGroupDialog(
            plan, name, self._unit, self, sweeps=group.sweeps if group else None
        )
        if dialog.exec() and plan.accepted:
            self.label_wanted.emit([step.node for step in plan.accepted], name)


def open_lookalikes_window(run, **kwargs) -> LookalikesWindow:
    """Open the window beside the Interpret tab, non-modal: it is a companion to the tree,
    selecting there the member clicked here, so the tree, its card and its clips stay usable
    while it is open. It is the page's child and closes with it."""
    window = LookalikesWindow(run, **kwargs)
    window.show()
    return window


class LabelGroupDialog(Dialog):
    """What labelling a group as one would do, before anything is written."""

    def __init__(
        self,
        plan,
        name: str,
        unit: str = "cluster",
        parent=None,
        *,
        sweeps: dict | None = None,
    ) -> None:
        from detail.api.cluster_lookalikes import SWEEP_WARNING
        from detail.api.cluster_lookalikes import plan_lines
        from detail.api.cluster_lookalikes import sweeps_in

        super().__init__(parent)
        self.setWindowTitle("Label as one")
        body = self.body()
        accepted = plan.accepted
        labelled, skipped = plan_lines(plan, unit, sweeps)
        units = unit if len(accepted) == 1 else f"{unit}s"
        heading = QLabel(
            f"<b>{len(accepted)} {units} will be labelled “{name}”.</b> One label, like one "
            "you typed, marked as given to a look-alike group."
        )
        heading.setWordWrap(True)
        body.addWidget(heading)
        if labelled:
            body.addWidget(_lines(labelled))
        if skipped:
            body.addWidget(QLabel(f"<b>Skipped {len(skipped)}</b>, and why:"))
            body.addWidget(_lines(skipped))
        if sweeps_in(plan, sweeps):
            warning = QLabel(SWEEP_WARNING.format(unit=unit))
            warning.setWordWrap(True)
            set_label_color(warning, WARN)
            body.addWidget(warning)
        note = QLabel(
            "Label edits have no undo: remove one afterwards with Remove label (U) on its "
            f"{unit}. The comparisons are kept."
        )
        note.setWordWrap(True)
        body.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText(f"Label {len(accepted)} {units}")
        set_available(
            ok, bool(accepted), "Nothing here can be labelled; the reasons are listed."
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)


def _swept_summary(swept, unit: str) -> str:
    """``#739, #931 +1 (12.0% of the run)`` for a member's sweeps; ``""`` for none."""
    if not swept:
        return ""
    shown = ", ".join(f"#{s.node}" for s in swept[:2])
    more = f" +{len(swept) - 2}" if len(swept) > 2 else ""
    return f"{shown}{more} ({sum(s.share for s in swept):.1%} of the run)"


def _lines(lines: list[str]) -> QLabel:
    label = QLabel("\n".join(f"• {line}" for line in lines))
    label.setWordWrap(True)
    return label
