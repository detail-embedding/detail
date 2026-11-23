"""The Plots view of a location-export result page (DESIGN_PLAN §16 "Location plots").

Controls on the left — one per ``LocationPlotView`` field, each built from the model so its
choices, bounds, default and help match the CLI — and the recording checklist; the figure on the
right in a zoomable ``ImageView``. The picture is the same matplotlib render an export writes
(``report/plot_render.render_png``), drawn off the UI thread through ``run_progress``: reading
the run's CSVs is the slow part, and ``LocationPlots`` keeps what it read, so changing a control
after the first draw only re-draws.

The run is opened on first show, never in ``__init__``, so a page that builds the result widget
and is never shown reads nothing.
"""

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.configs.location_plots import LocationPlotView
from detail.gui._pydantic_form import (
    any_field_value,
    any_field_widget,
    field_advanced,
    field_meta,
    field_slider,
    optional_spin,
    optional_value,
)
from detail.gui.forms import ConfigForm
from detail.gui.grip_splitter import GripSplitter
from detail.gui.image_view import ImageView
from detail.gui.sizing import set_available

#: Fields drawn as a camera slider rather than a box: the 3-D view is *turned*, not typed.
_SLIDERS = ("elev", "azim")
_PATHS = ("trace", "trace3d")
_POSITIONS = ("trace", "trace3d", "heatmap", "speed")
_BODY = ("body", "body_hist")
#: Which plot kinds each field means something for; a field not listed applies to every kind.
_APPLIES_TO = {
    "source": _POSITIONS,
    "body_measure": _BODY,
    "color_by": _PATHS,
    "posture_every_s": ("trace",),
    "speed_clip_pct": _PATHS,
    "max_points": _PATHS,
    "smooth_s": _PATHS + ("speed",),
    "jump_speed": _PATHS + ("speed",),
    "bins": ("heatmap",),
    "hist_bins": ("body_hist",),
    "time_bin_s": ("speed", "body"),
    "elev": ("trace3d",),
    "azim": ("trace3d",),
}
#: Rows that need the body, which an export made before it was measured does not have.
_NEEDS_BODY = ("body_measure", "posture_every_s")
#: The group-by combo's "no grouping" entry.
EACH_RECORDING = "(each recording)"
#: What the body-outline box reads when it is left unset.
NO_OUTLINE_TEXT = "none"
#: How long the controls must sit still before a redraw, so a slider sweep draws once.
REDRAW_DEBOUNCE_MS = 250
#: Formats an export can be saved as, as the save dialog lists them.
EXPORT_FILTERS = (
    "PNG image (*.png);;SVG image (*.svg);;PDF (*.pdf);;Interactive HTML (*.html)"
)


class LocationPlotsPanel(QWidget):
    """Plot a location-export run per recording or by a Recording Information column."""

    def __init__(self, experiment, run_id: str, parent=None) -> None:
        super().__init__(parent)
        self._experiment = experiment
        self._run_id = run_id
        self._plots = None
        self._opened = False

        controls = QWidget()
        column = QVBoxLayout(controls)
        column.setContentsMargins(0, 0, 0, 0)
        form_host = QWidget()
        self._form = ConfigForm(form_host, LocationPlotView)
        self._widgets: dict[str, QWidget] = {}
        for name in LocationPlotView.model_fields:
            meta = field_meta(LocationPlotView, name)
            if name == "group_by":
                widget = QComboBox()
                widget.addItem(EACH_RECORDING, None)
                widget.setToolTip(meta["help"])
                widget.currentIndexChanged.connect(self._schedule)
            elif name == "posture_every_s":
                widget = optional_spin(LocationPlotView, name, NO_OUTLINE_TEXT)
                _connect(widget, self._schedule)
            elif name in _SLIDERS:
                widget = field_slider(LocationPlotView, name)
                widget.set_value(LocationPlotView.model_fields[name].default)
                widget.threshold_changed.connect(self._schedule)
            else:
                widget = any_field_widget(LocationPlotView, name)
                _connect(widget, self._schedule)
            self._widgets[name] = widget
            self._form.addRow(
                f"{meta['label']}:",
                widget,
                advanced=field_advanced(LocationPlotView, name),
            )
        self._form.finish()
        column.addWidget(form_host)
        # Whether the export measured the body; until the run is read, assume it did.
        self._has_body = True

        column.addWidget(QLabel("Recordings:"))
        self._recordings = QListWidget()
        self._recordings.itemChanged.connect(self._schedule)
        column.addWidget(self._recordings, 1)
        self._export_btn = QPushButton("Export image…")
        self._export_btn.clicked.connect(self._export)
        set_available(self._export_btn, False, "Opens once the run has been read.")
        column.addWidget(self._export_btn)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        column.addWidget(self._status)

        self._picture = ImageView(zoomable=True)
        # The controls fold by their grip to a strip reading "Plot settings"; the picture
        # does not.
        split = GripSplitter(Qt.Orientation.Horizontal, key="location_plots")
        split.addWidget(controls)
        split.addWidget(self._picture)
        split.setStretchFactor(1, 1)
        split.set_foldable(0, "Plot settings")
        if not split.remember():
            split.setSizes([300, 700])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REDRAW_DEBOUNCE_MS)
        self._timer.timeout.connect(self._draw)
        self._sync_rows()

    # -- opening --------------------------------------------------------------- #
    def showEvent(self, event) -> None:  # noqa: N802 - Qt's own name
        super().showEvent(event)
        if not self._opened:
            self._opened = True
            from detail.gui.busy import run_progress

            experiment, run_id = self._experiment, self._run_id

            def open_run(_emit):
                plots = experiment.location_plots(run_id)
                return plots, plots.body_measured()

            run_progress(
                self,
                open_run,
                self._on_opened,
                label="Reading the run…",
                on_error=self._failed,
            )

    def _on_opened(self, opened) -> None:
        plots, has_body = opened
        self._plots = plots
        self._has_body = has_body
        self._sync_rows()
        names = plots.names()
        self._recordings.blockSignals(True)
        for sid in plots.recording_ids:
            item = QListWidgetItem(names[sid])
            item.setData(Qt.ItemDataRole.UserRole, sid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._recordings.addItem(item)
        self._recordings.blockSignals(False)
        combo = self._widgets["group_by"]
        combo.blockSignals(True)
        for column in plots.group_columns:
            combo.addItem(column, column)
        combo.blockSignals(False)
        set_available(self._export_btn, True, "")
        self._draw()

    # -- the view -------------------------------------------------------------- #
    def view(self) -> LocationPlotView:
        """The controls as a ``LocationPlotView`` — every field, read from its widget."""
        values = {}
        for name, widget in self._widgets.items():
            if name == "group_by":
                values[name] = widget.currentData()
            elif name == "posture_every_s":
                values[name] = optional_value(widget)
            elif name in _SLIDERS:
                values[name] = widget.value()
            else:
                values[name] = any_field_value(LocationPlotView, name, widget)
        return LocationPlotView(**values)

    def selected_recordings(self) -> list[str]:
        """The checked recordings, in run order."""
        return [
            self._recordings.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._recordings.count())
            if self._recordings.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _sync_rows(self) -> None:
        kind = any_field_value(LocationPlotView, "kind", self._widgets["kind"])
        main = self._form.main
        for name in set(_APPLIES_TO) | set(_NEEDS_BODY):
            widget = self._widgets[name]
            shown = kind in _APPLIES_TO.get(name, (kind,)) and (
                self._has_body or name not in _NEEDS_BODY
            )
            if main.indexOf(widget) >= 0:
                main.setRowVisible(widget, shown)
            else:
                widget.setEnabled(shown)

    def _schedule(self, *_args) -> None:
        self._sync_rows()
        if self._plots is not None:
            self._timer.start()

    # -- drawing --------------------------------------------------------------- #
    def _draw(self) -> None:
        if self._plots is None:
            return
        chosen = self.selected_recordings()
        if not chosen:
            self._picture.clear_frame("Tick at least one recording to plot.")
            return
        from detail.gui import plot_theme
        from detail.gui.busy import run_progress
        from detail.report import plot_render

        view, plots = self.view(), self._plots
        theme = plot_theme.current().name

        def work(emit):
            spec = plots.build(view, chosen, progress=emit)
            return plot_render.render_png(spec, theme=theme)

        self._status.setText("")
        run_progress(
            self._picture,
            work,
            self._show,
            label="Drawing…",
            on_error=self._failed,
            supersedes="location-plot",
        )

    def _show(self, png: bytes) -> None:
        pixmap = QPixmap()
        pixmap.loadFromData(png, "PNG")
        self._picture.set_frame(pixmap)

    def _failed(self, exc: BaseException) -> None:
        self._status.setText(str(exc))

    def _export(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export plot",
            f"{self._run_id}_{self.view().kind}.png",
            EXPORT_FILTERS,
        )
        if not path:
            return
        self.export_to(path)

    def export_to(self, path: str) -> None:
        """Write the current plot to ``path`` (its suffix picks the format), off the UI thread."""
        from detail.gui.busy import run_progress

        target = Path(path)
        fmt = target.suffix.lstrip(".").lower() or "png"
        view, plots, chosen = self.view(), self._plots, self.selected_recordings()
        run_progress(
            self,
            lambda emit: plots.export(
                view, target.with_suffix(""), chosen, formats=(fmt,), progress=emit
            ),
            lambda paths: self._status.setText(f"Wrote {paths[0]}"),
            label="Exporting…",
            on_error=self._failed,
        )


def _connect(widget: QWidget, slot) -> None:
    """Call ``slot`` when a generated field widget changes, whatever its shape."""
    for signal in ("currentIndexChanged", "valueChanged", "toggled"):
        if hasattr(widget, signal):
            getattr(widget, signal).connect(slot)
            return
    raise TypeError(f"no change signal on {type(widget).__name__}")
