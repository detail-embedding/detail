"""What AUDIT found about the selected cluster, and what to do about it — on INTERPRET (§19, §22).

A user picks clusters that look pure; AUDIT reviews every clip of each one and gives its exact
precision (``review_by_cluster``, shown once the cluster is reviewed; the hidden pose method still
reads its one-by-one audit, a sample with a 95% confidence interval). This block carries the figure
back to where the cluster was picked, with the three edits a reader may decide it calls for.
Whether it does is the reader's call; the block gives no verdict. The figure reads as AUDIT reads
it (``storage/audit_figures.py``): the exact share once every clip is reviewed; for a sample, no
estimate below 8 audited.

- **Move to…** another label, when its clips were mostly something else;
- **Remove from the label**, when they were nothing in particular;
- **Look at its halves**, when it may be two things — one pure half worth keeping.

One line in the label bar, under the cluster's name: the figure, what its clips were, and the
three edits in one **Act on it ▾** menu, so the audit costs one line and not a row of buttons.
Hidden on a cluster nobody has audited. The edits are disabled, with the reason, wherever
the run cannot change (a snapshot, a finished run).
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from detail.storage import audit_figures
from detail.storage import behavior_audit as audit


class NodeAuditBlock(QWidget):
    """The selected cluster's audited precision, with Move to… / Remove / Look at its halves."""

    move_wanted = pyqtSignal(int, str)  # node, label name
    remove_wanted = pyqtSignal(int)
    halves_wanted = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        row = QHBoxLayout()
        row.addWidget(QLabel("<b>AUDIT</b>"))
        self._figure = QLabel("")
        row.addWidget(self._figure)
        self._answers = QLabel("")
        row.addWidget(self._answers, 1)
        self._act = QToolButton()
        self._act.setText("Act on it ▾")
        self._act.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self._act)
        menu.setToolTipsVisible(True)
        self._move_menu = menu.addMenu("Move to…")
        self._move_menu.setToolTipsVisible(True)  # a refused label says why
        self._move = self._move_menu.menuAction()
        self._remove = menu.addAction(
            "Remove from the label", lambda: self.remove_wanted.emit(self._node)
        )
        self._halves = menu.addAction(
            "Look at its halves", lambda: self.halves_wanted.emit(self._node)
        )
        self._act.setMenu(menu)
        for action in (self._move, self._remove, self._halves):
            # Off until a figure is shown, with what the run allows.
            action.setEnabled(False)
        row.addWidget(self._act)
        layout.addLayout(row)
        self._why = QLabel("")
        self._why.setWordWrap(True)
        layout.addWidget(self._why)
        self._node = -1
        self.setVisible(False)

    def show_node(
        self,
        node: int,
        row: dict | None,
        choices: list[tuple[str, int, str | None]],
        *,
        has_halves: bool,
        editable_reason: str,
    ) -> None:
        """Show ``row`` (an ``audit_by_cluster`` row) for ``node``; hide without one.

        ``choices`` are the labels the cluster could join, each with the reason it may not
        (``label_panel.label_choices``); ``editable_reason`` is empty where the run can change,
        else why it cannot. A sample's CI width is read against the default goal
        (``behavior_audit.DEFAULT_GOAL_WIDTH``).
        """
        self._node = int(node)
        estimate = (row or {}).get("estimate") or {}
        if row is None or not estimate.get("n"):
            self.setVisible(False)
            return
        value, interval = audit_figures.cells(estimate)
        width = audit_figures.width(estimate)
        if audit_figures.complete(estimate):
            # A census: the share is a count, with no interval to narrow.
            parts = [f"precision {value}", f"all {estimate['n']} clips reviewed"]
            width = None
        else:
            parts = [f"precision {value}", f"95% CI {interval.replace(' – ', '–')}"]
            parts.append(f"{estimate['n']} of {row.get('size', '?')} reviewed")
        if width is not None:
            goal = audit.DEFAULT_GOAL_WIDTH
            reached = ", reached" if audit_figures.goal_reached(estimate, goal) else ""
            parts.append(f"CI width {width:.2f} (goal {goal:.2f}{reached})")
        self._figure.setText(" · ".join(parts))
        self._figure.setToolTip("\n".join(audit_figures.notes(estimate)))
        answers = row.get("answers") or {}
        top = sorted(answers.items(), key=lambda kv: -kv[1])[:3]
        self._answers.setText(
            "· its clips were: " + ", ".join(f"{text} ({n})" for text, n in top)
            if top
            else ""
        )
        self._move_menu.clear()
        for name, _clusters, why_not in choices:
            action = self._move_menu.addAction(
                name, lambda n=name: self.move_wanted.emit(self._node, n)
            )
            action.setEnabled(why_not is None)
            action.setToolTip(why_not or "")
        editable = not editable_reason
        self._move.setEnabled(editable and not self._move_menu.isEmpty())
        self._remove.setEnabled(editable)
        self._halves.setEnabled(has_halves)
        self._halves.setToolTip("" if has_halves else "This cluster has no halves.")
        self._why.setText(editable_reason)
        self._why.setVisible(bool(editable_reason))
        self.setVisible(True)
