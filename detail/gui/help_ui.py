"""In-app help popover + the ? button that opens it (PR3).

A `?` (`help_button`) pops a small panel anchored to itself with the topic's doc prose
(read from the shipped markdown — never authored here) and an "Open full docs ↗" escape
hatch. It's a top-level popup, so it floats above whatever opened it — including a modal
launch dialog, where a main-window dock would be unreachable.
"""

from __future__ import annotations

from detail import docs_access
from detail import help as help_mod
from detail.gui.busy import run_progress
from detail.gui.qt import (
    QDesktopServices,
    QFrame,
    QGuiApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPoint,
    QPushButton,
    QToolButton,
    QUrl,
    QVBoxLayout,
    QWidget,
    Qt,
)

_POPOVER_WIDTH = 380


def build_and_open_docs(parent: QWidget, topic: help_mod.HelpTopic) -> None:
    """Open a topic's docs page in the browser, building the offline site first if needed.

    The build (mkdocs, seconds on first run or after a doc change) runs off the UI thread
    behind a "Building documentation…" progress bar over ``parent`` (its window), so the
    click isn't a silent wait. Opens the localhost URL so the browser honors the
    ``#anchor``; errors surface on ``parent``."""

    def open_now(_result: object = None) -> None:
        QDesktopServices.openUrl(QUrl(docs_access.local_url(topic.page, topic.anchor)))

    if not docs_access.needs_build():
        open_now()
        return

    run_progress(
        parent,
        lambda emit: docs_access.ensure_site_built(progress=emit),
        open_now,
        label="Building documentation…",
        on_error=lambda exc: QMessageBox.critical(
            parent, "Documentation", f"Could not build the documentation:\n{exc}"
        ),
    )


class HelpPopover(QFrame):
    """A small popup, anchored to a `?`, rendering a topic's doc prose + Open-full-docs.

    A ``Qt.Popup`` top-level: it floats above a modal launch dialog (a main-window dock
    can't) and dismisses itself on an outside click or Esc.
    """

    def __init__(self, topic: help_mod.HelpTopic, anchor: QWidget):
        super().__init__(anchor.window(), Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._topic = topic
        self._host = anchor.window()  # durable owner for an off-thread doc build
        layout = QVBoxLayout(self)
        self._label = QLabel()
        self._label.setTextFormat(Qt.TextFormat.MarkdownText)
        self._label.setWordWrap(True)  # wraps → no cut-off
        self._label.setMaximumWidth(_POPOVER_WIDTH)
        self._label.setOpenExternalLinks(False)
        self._label.linkActivated.connect(self._on_link)
        self._label.setText(help_mod.body_for(topic) or "See the full documentation.")
        layout.addWidget(self._label)
        open_btn = QPushButton("Open full docs ↗")
        open_btn.clicked.connect(self._open_full)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(open_btn)
        layout.addLayout(row)

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

    def _open_full(self) -> None:
        topic, host = self._topic, self._host
        self.close()  # dismiss before the (possibly slow) build so input isn't grabbed
        build_and_open_docs(host, topic)

    def _on_link(self, href: str) -> None:
        if href.startswith("http"):
            QDesktopServices.openUrl(QUrl(href))


def _topic_button(topic: help_mod.HelpTopic) -> QToolButton:
    """A small ? that pops a help panel for a docs ``topic``. The prose comes from the docs
    (``detail/help.py``) — no help text is authored here."""
    button = QToolButton()
    button.setText("?")
    button.setAutoRaise(True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setToolTip("Click for help")

    def open_popover() -> None:
        button._popover = HelpPopover(topic, button)  # keep a reference
        button._popover.show_at(button)

    button.clicked.connect(open_popover)
    return button


def help_button(topic_key: str) -> QToolButton:
    """A ? for a HELP concept/field topic key (see ``detail/help.py``). Lives here (not in
    ``pages``) so any form/module can add one without an import cycle."""
    return _topic_button(help_mod.HELP[topic_key])


def field_help_button(module_id: str, field: str) -> QToolButton:
    """A ? for a config field, opening its entry on the module page (``modules/<id>#<field>``)."""
    return _topic_button(help_mod.field_topic(module_id, field))


def _fill_label(row: QWidget, text: str, button: QToolButton) -> QWidget:
    box = QHBoxLayout(row)
    box.setContentsMargins(0, 0, 0, 0)
    box.addWidget(QLabel(text))
    box.addWidget(button)
    box.addStretch()
    return row


def label_with(text: str, button: QToolButton) -> QWidget:
    """A form label ``text`` with a trailing ? ``button`` — pass as the ``label`` to
    ``QFormLayout.addRow`` (so a row-hide via the field still hides the label + its ?).
    """
    return _fill_label(QWidget(), text, button)


class FieldLabel(QWidget):
    """A form label that remembers *which config field* its row is for.

    The identity is what lets ``gui/forms.py::ConfigForm`` route a row to the advanced
    section without the form deciding row by row: the label already knows the field, so the
    model's ``advanced`` flag is enough. Every labeled row of every config form is built
    through ``field_label`` (the help-coverage guard enforces it), so the coverage is free.
    """

    def __init__(self, module_id: str, field: str) -> None:
        super().__init__()
        self.module_id = module_id
        self.field = field


def field_label(module_id: str, field: str, text: str) -> QWidget:
    """A form label with a trailing field ? for ``modules/<id>#<field>``."""
    return _fill_label(
        FieldLabel(module_id, field), text, field_help_button(module_id, field)
    )
