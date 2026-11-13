"""The Plots panel of a location-export result page (DESIGN_PLAN §16 "Location plots").

Controls on the left — one per ``LocationPlotView`` field, each built from the model so its
choices, bounds, default and help match the CLI — and the recording checklist; the figure on the
right in a zoomable ``ImageView``. The picture is the same matplotlib render an export writes
(``report/plot_render.render_png``), drawn off the UI thread through ``run_progress``: reading
the run's CSVs is the slow part, and ``LocationPlots`` keeps what it read, so changing a control
after the first draw only re-draws.

The run is opened on first show, never in ``__init__``, so a page that builds the result widget
and is never shown reads nothing. Opening reads every recording's rows, behind a progress bar:
the **Animals** checklist needs every label in the run (a run without labels names its animals
by track number, and only the rows say which tracks there are), and the first plot needs the rows
anyway.
"""

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

from detail.analysis.location_plots import CENTROID_SOURCES
from detail.configs.location_plots import LocationPlotView
from detail.gui._pydantic_form import (
    any_field_value,
    any_field_widget,
    field_advanced,
    field_meta,
    field_slider,
)
from detail.gui.forms import ConfigForm
from detail.gui.grip_splitter import GripSplitter
from detail.gui.image_view import ImageView
from detail.gui.recording_filter import with_filter
from detail.gui.sizing import set_available

#: Fields drawn as a camera slider rather than a box: the 3-D view is *turned*, not typed.
_SLIDERS = ("elev", "azim")
#: Fields with a hand-built widget: choices the run supplies (a metadata column, its animals,
#: its keypoints, its pairs).
_HAND_BUILT = ("group_by", "subjects", "source", "points", "pairs")
#: Checklists whose every row ticked means "all" (the model's None).
_ALL_OR_SOME = ("subjects", "pairs")
_PATHS = ("trace", "trace3d")
_POSITIONS = ("trace", "trace3d", "heatmap", "speed")
_DISTANCES = ("distance", "distance_hist")
#: Which plot kinds each field means something for; a field not listed applies to every kind.
_APPLIES_TO = {
    "source": _POSITIONS,
    "points": _PATHS,
    "units": _POSITIONS + _DISTANCES,
    "color_by": _PATHS,
    "speed_clip_pct": _PATHS,
    "max_points": _PATHS,
    "bins": ("heatmap",),
    "time_bin_s": ("speed", "distance"),
    "pairs": _DISTANCES,
    "distance_by": _DISTANCES,
    "hist_bins": ("distance_hist",),
    "elev": ("trace3d",),
    "azim": ("trace3d",),
}
#: The animal checklist's height: a few rows, since a recording rarely holds more animals.
_ANIMAL_LIST_PX = 96
#: The group-by combo's "no grouping" entry.
EACH_RECORDING = "(each recording)"
#: How long the controls must sit still before a redraw, so a slider sweep draws once.
REDRAW_DEBOUNCE_MS = 250


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
            elif name in ("subjects", "points", "pairs"):
                widget = QListWidget()
                widget.setToolTip(meta["help"])
                widget.setMaximumHeight(_ANIMAL_LIST_PX)
                widget.itemChanged.connect(self._schedule)
            elif name == "source":
                widget = QComboBox()
                for centroid in CENTROID_SOURCES:
                    widget.addItem(centroid, centroid)
                widget.setToolTip(meta["help"])
                widget.currentIndexChanged.connect(self._schedule)
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
        # Rows the run itself has nothing for (one animal, no keypoints, no pair) stay hidden
        # whatever the plot kind; filled in once the run is read.
        self._available = {"subjects": False, "split_by_subject": False}
        self._available.update(points=False, pairs=False)

        column.addWidget(QLabel("Recordings:"))
        self._recordings = QListWidget()
        self._recordings.itemChanged.connect(self._schedule)
        column.addWidget(
            with_filter(self._recordings, experiment, select_buttons=True), 1
        )
        self._export_btn = QPushButton("Export image…")
        self._export_btn.clicked.connect(self._export)
        set_available(self._export_btn, False, "Opens once the run has been read.")
        column.addWidget(self._export_btn)
        self._montage_btn = QPushButton("Montage…")
        self._montage_btn.setToolTip(
            "Play the ticked recordings side by side, every ticked animal in its colour "
            "(one recording is a preview), and export it."
        )
        self._montage_btn.clicked.connect(self.open_montage)
        set_available(self._montage_btn, False, "Opens once the run has been read.")
        column.addWidget(self._montage_btn)
        self._status = QLabel("")
        self._status.setWordWrap(True)
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
        # The picture is the point of the tab; without a floor the page's layout gives it
        # whatever height the controls column happens to need.
        self.setMinimumHeight(560)

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

            def open_run(emit):
                plots = experiment.location_plots(run_id)
                animals = plots.subject_choices(progress=emit)
                return plots, animals, plots.pair_choices(), plots.keypoint_names

            run_progress(
                self,
                open_run,
                self._on_opened,
                label="Reading the run…",
                on_error=self._failed,
            )

    def _on_opened(self, opened) -> None:
        plots, animals, pairs, keypoint_names = opened
        self._plots = plots
        _fill_checklist(self._widgets["subjects"], animals, ticked=True)
        _fill_checklist(self._widgets["pairs"], pairs, ticked=True)
        _fill_checklist(self._widgets["points"], keypoint_names, ticked=False)
        source = self._widgets["source"]
        source.blockSignals(True)
        for point in keypoint_names:
            source.addItem(point, point)
        source.blockSignals(False)
        # One animal everywhere leaves nothing to choose or group by; no keypoints, no points.
        self._available.update(
            subjects=len(animals) > 1,
            split_by_subject=len(animals) > 1,
            points=bool(keypoint_names),
            pairs=bool(pairs),
        )
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
        set_available(self._montage_btn, True, "")
        self._draw()

    # -- the view -------------------------------------------------------------- #
    def view(self) -> LocationPlotView:
        """The controls as a ``LocationPlotView`` — every field, read from its widget."""
        values = {}
        for name, widget in self._widgets.items():
            if name in ("group_by", "source"):
                values[name] = widget.currentData()
            elif name in _ALL_OR_SOME:
                values[name] = _all_or_ticked(widget)
            elif name == "points":
                values[name] = _ticked(widget) or None
            elif name in _SLIDERS:
                values[name] = widget.value()
            else:
                values[name] = any_field_value(LocationPlotView, name, widget)
        return LocationPlotView(**values)

    def _chosen_animals(self) -> list[str] | None:
        """The ticked animal labels, or None when every one is ticked (the model's "all")."""
        return _all_or_ticked(self._widgets["subjects"])

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
        for name in set(_APPLIES_TO) | set(self._available):
            widget = self._widgets[name]
            shown = kind in _APPLIES_TO.get(name, (kind,)) and self._available.get(
                name, True
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
        if self._chosen_animals() == []:
            self._picture.clear_frame("Tick at least one animal to plot.")
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

    def open_montage(self) -> None:
        """Open the montage window over the ticked recordings and animals (§16 "Preview and
        montage"), modal like the cluster montage."""
        from detail.gui.location_montage import LocationMontageDialog

        chosen = self.selected_recordings()
        if not chosen:
            self._status.setText("Tick at least one recording to play.")
            return
        LocationMontageDialog(
            self._experiment, self._plots, chosen, self._chosen_animals(), parent=self
        ).exec()

    def _export(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export plot",
            f"{self._run_id}_{self.view().kind}.png",
            "PNG image (*.png);;SVG image (*.svg);;PDF (*.pdf);;"
            "Interactive HTML (*.html)",
        )
        if not path:
            return
        self.export_to(path)

    def export_to(self, path: str) -> None:
        """Write the current plot to ``path`` (its suffix picks the format), off the UI thread."""
        from pathlib import Path

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


def _fill_checklist(listing: QListWidget, labels: list[str], *, ticked: bool) -> None:
    """Fill a checklist with ``labels``, every row ticked or none, without a redraw per row."""
    listing.blockSignals(True)
    for label in labels:
        item = QListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked)
        listing.addItem(item)
    listing.blockSignals(False)


def _ticked(listing: QListWidget) -> list[str]:
    return [
        listing.item(i).text()
        for i in range(listing.count())
        if listing.item(i).checkState() == Qt.CheckState.Checked
    ]


def _all_or_ticked(listing: QListWidget) -> list[str] | None:
    """The ticked rows, or None when every one is ticked (the model's "all")."""
    ticked = _ticked(listing)
    return None if len(ticked) == listing.count() else ticked


def _connect(widget: QWidget, slot) -> None:
    """Call ``slot`` when a generated field widget changes, whatever its shape."""
    for signal in ("currentIndexChanged", "valueChanged", "toggled"):
        if hasattr(widget, signal):
            getattr(widget, signal).connect(slot)
            return
    raise TypeError(f"no change signal on {type(widget).__name__}")
