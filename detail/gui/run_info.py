"""The meta-information window: a run's YAML files, one tab each, with Export.

Shared by every run page that has meta-information to show (today, every analysis run's page,
whatever its module), so it takes the files already read (`api/run_info.RunInfo`) rather than a
run: the caller reads them off the UI thread, and this window only shows them. Export writes the
files it shows to one zip (`api/run_info.export_run_info`), off the UI thread too.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QTabWidget,
)

from detail.api import run_info as api_run_info
from detail.gui.busy import run_busy
from detail.gui.dialog import Dialog
from detail.gui.sizing import mono_font

#: The most lines a file asks room for; a longer one scrolls inside its tab. The file scrolls
#: rather than the window, so the tabs stay in view (an_17's ``analysis.yaml`` is 230 lines).
#: Small enough that the window fits an 800-pixel-high screen unscrolled; the text grows when
#: the window is made taller.
MAX_LINES_SHOWN = 24


class _FileText(QPlainTextEdit):
    """One file's text, read-only and unwrapped, as wide as its longest line and as tall as
    the file up to :data:`MAX_LINES_SHOWN` lines; it grows with the window."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setReadOnly(True)
        self.setFont(mono_font())
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt's name
        metrics = QFontMetrics(self.font())
        lines = self.toPlainText().splitlines() or [""]
        margin = 2 * (self.frameWidth() + int(self.document().documentMargin()))
        bar = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        width = max(metrics.horizontalAdvance(line) for line in lines)
        shown = min(len(lines), MAX_LINES_SHOWN)
        return QSize(width + margin + bar, metrics.lineSpacing() * shown + margin + bar)


class RunInfoWindow(Dialog):
    """A run's meta-information files, one tab each, and **Export…** to a zip."""

    def __init__(self, run_info: api_run_info.RunInfo, parent=None) -> None:
        super().__init__(parent)
        self._info = run_info
        self.setWindowTitle(f"Meta-information: {run_info.run_id}")
        body = self.body()
        if run_info.files:
            intro = QLabel(
                "The files this run keeps about itself, as they are on disk: its record and "
                "every setting it was launched with, the module's own settings, and where it ran."
            )
            intro.setWordWrap(True)
            body.addWidget(intro)
            self._tabs = QTabWidget()
            # Every file's name in view: the window is as wide as the tab bar, never arrows.
            self._tabs.setUsesScrollButtons(False)
            for f in run_info.files:
                self._tabs.addTab(_FileText(f.text), f.name)
            body.addWidget(self._tabs)
        else:
            self._tabs = None
            body.addWidget(QLabel(f"{run_info.run_id} keeps no YAML files."))
        self._status = QLabel("")
        self._status.setWordWrap(True)
        body.addWidget(self._status)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self._export = QPushButton("Export…")
        self._export.setToolTip("Save every file shown here to one zip.")
        self._export.setEnabled(bool(run_info.files))
        if not run_info.files:
            self._export.setToolTip("There are no files to export.")
        self._export.clicked.connect(self._choose_export)
        buttons.addButton(self._export, QDialogButtonBox.ButtonRole.ActionRole)
        self.set_buttons(buttons)

    def _choose_export(self) -> None:
        suggested = str(Path.home() / f"{self._info.run_id}_meta-information.zip")
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export meta-information", suggested, "Zip files (*.zip)"
        )
        if path:
            self.export_to(Path(path))

    def export_to(self, path: Path) -> None:
        """Write the shown files to ``path`` (``.zip`` added if missing), off the UI thread."""
        if path.suffix.lower() != ".zip":
            path = path.with_name(path.name + ".zip")
        info = self._info
        run_busy(
            self,
            lambda: api_run_info.export_run_info(info, path),
            self._exported,
            label="Exporting…",
            supersedes="export",
        )

    def _exported(self, path) -> None:
        self._status.setText(f"Exported to {path}")
