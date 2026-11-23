"""Version History: which build is running, whether it is current, and what changed.

Nothing here is maintained by hand. The build facts and the history both come from git
(``core/buildinfo.py``): the checked-out commit and its date say how current a copy is, and the
first-parent log of HEAD — on ``main``, one merged PR per entry, shown by its commit subject —
is the list of what changed. "Check for updates" is the one network call, made only when clicked.
"""

from __future__ import annotations

from detail.core import buildinfo
from detail.gui.busy import run_busy
from detail.gui.qt import (
    QFrame,
    QGuiApplication,
    QHBoxLayout,
    QLabel,
    QPoint,
    QPushButton,
    QScrollArea,
    QSize,
    QVBoxLayout,
    QWidget,
    Qt,
)

_POPOVER_WIDTH = 520
_POPOVER_MAX_HEIGHT = 420


class BuildLabel(QLabel):
    """The running build, for the right end of the mode banner.

    Reports no minimum width and shortens to fit: a label that is always present would
    otherwise raise the whole window's minimum width by its full text (572px to 732px,
    measured). It drops whole parts — commit, then version (``BuildInfo.labels``) — so the date
    is the last thing to go, and only then elides.
    """

    def __init__(self) -> None:
        super().__init__("")
        self._labels: list[str] = [""]
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    @property
    def _full(self) -> str:
        return self._labels[0]

    def set_build(self, info: buildinfo.BuildInfo) -> None:
        self._labels = info.labels()
        self.setToolTip(f"The build of detail that is running: {self._full}")
        self.updateGeometry()
        self._fit()

    def build_text(self) -> str:
        """The full label, however much of it currently fits."""
        return self._full

    def sizeHint(self) -> QSize:
        # From the full text, not the elided one on screen — otherwise each elision would
        # shrink the hint and the layout would hand it less room the next time.
        margins = self.contentsMargins()
        width = (
            self.fontMetrics().horizontalAdvance(self._full)
            + margins.left()
            + margins.right()
            + 2 * self.margin()
        )
        return QSize(width, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        metrics = self.fontMetrics()
        room = max(self.width(), 1)
        for text in self._labels:
            if metrics.horizontalAdvance(text) <= room:
                self.setText(text)
                return
        self.setText(
            metrics.elidedText(self._labels[-1], Qt.TextElideMode.ElideRight, room)
        )


def _describe(info: buildinfo.BuildInfo | None) -> str:
    return "Reading build…" if info is None else info.describe()


class VersionHistoryPopover(QFrame):
    """A popup, anchored to the Version History button: build, update check, history.

    A ``Qt.Popup`` top-level, the same shape as ``help_ui.HelpPopover``: it floats above a
    modal dialog and dismisses itself on an outside click or Esc.
    """

    def __init__(self, anchor: QWidget, info: buildinfo.BuildInfo | None = None):
        super().__init__(anchor.window(), Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        title = QLabel("Version History")
        bold = title.font()
        bold.setBold(True)
        title.setFont(bold)
        layout.addWidget(title)

        self._build = QLabel(_describe(info))
        self._build.setWordWrap(True)
        self._build.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._build)

        update_row = QHBoxLayout()
        self._check = QPushButton("Check for updates")
        self._check.setToolTip(f"Fetch {buildinfo.UPSTREAM_REMOTE}/main and compare")
        self._check.clicked.connect(self._check_for_updates)
        self._update_status = QLabel("")
        self._update_status.setWordWrap(True)
        update_row.addWidget(self._check)
        update_row.addWidget(self._update_status, 1)
        layout.addLayout(update_row)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(self._body)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(_POPOVER_WIDTH)
        scroll.setMinimumHeight(_POPOVER_MAX_HEIGHT // 2)
        scroll.setMaximumHeight(_POPOVER_MAX_HEIGHT)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(scroll)

        if info is None:
            run_busy(
                self, buildinfo.build_info, self._show_build, label="Reading build…"
            )
        run_busy(
            self._body, buildinfo.history, self._show_history, label="Reading history…"
        )

    def _show_build(self, info: buildinfo.BuildInfo) -> None:
        self._build.setText(_describe(info))

    def _show_history(self, entries: list[buildinfo.HistoryEntry]) -> None:
        if not entries:
            self._add_row("No history — this copy of detail is not a git checkout.")
            return
        for entry in entries:
            row = self._add_row(f"{entry.date}   {entry.subject}")
            row.setToolTip(f"commit {entry.commit}")

    def _add_row(self, text: str) -> QLabel:
        row = QLabel(text)
        row.setWordWrap(True)
        row.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._body_layout.insertWidget(self._body_layout.count() - 1, row)
        return row

    def _check_for_updates(self) -> None:
        self._check.setEnabled(False)
        self._update_status.setText("Checking…")
        run_busy(
            self,
            buildinfo.check_for_updates,
            self._show_update,
            label="Checking for updates…",
            on_error=self._show_update_error,
        )

    def _show_update(self, status: buildinfo.UpdateStatus) -> None:
        self._check.setEnabled(True)
        self._update_status.setText(status.describe())

    def _show_update_error(self, exc: BaseException) -> None:
        self._check.setEnabled(True)
        self._update_status.setText(str(exc))

    def show_at(self, anchor: QWidget) -> None:
        """Reveal just below the anchor, clamped to the screen (flipped above if it won't fit)."""
        self.adjustSize()
        pos = anchor.mapToGlobal(QPoint(0, anchor.height()))
        area = (
            QGuiApplication.screenAt(pos) or QGuiApplication.primaryScreen()
        ).availableGeometry()
        x = min(pos.x(), area.right() - self.width())
        y = pos.y()
        if y + self.height() > area.bottom():
            y = anchor.mapToGlobal(QPoint(0, 0)).y() - self.height()
        self.move(max(area.left(), x), max(area.top(), y))
        self.show()
