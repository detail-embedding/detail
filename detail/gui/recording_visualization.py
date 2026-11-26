"""Recording Embedding and Location Visualization — INTERPRET's **Visualize recordings…** window
(DESIGN_PLAN §22 "Recording visualization").

One recording of the clustering run, whole or a range, played beside its path through pose space,
its ethogram and (optionally) Location Export's path over time, on black or white. The picture is
drawn by ``api/recording_visualization`` on the scrubber's decode thread, so scrubbing and playing
never block the window; loading a recording runs through ``run_progress``. Cluster colours come
from the run's names or from a cut drawn across its tree (:class:`TreeCut`), at most eight.
**Save visualization** keeps the recipe in the run; **Export…** writes it, or a montage of saved
recipes from any experiment (:class:`ExportVisualizationDialog`).

Every control's bounds, default and help come from ``configs/recording_visualization``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from detail.analysis import recording_visualization as rv
from detail.configs.recording_visualization import (
    MAX_COLOURED,
    RecordingVisualizationConfig,
    RecordingVisualizationExport,
)
from detail.gui import frame_source, plot_theme
from detail.gui._pydantic_form import (
    any_field_value,
    any_field_widget,
    field_meta,
    set_any_field_value,
)
from detail.gui.busy import run_busy, run_progress
from detail.gui.dialog import Dialog
from detail.gui.forms import ConfigForm
from detail.gui.image_view import ImageView, PictureWithControls
from detail.gui.run_preview import _to_pixmap
from detail.gui.scrubber import FrameScrubber
from detail.gui.sizing import lines, set_available

#: The preview is drawn at this size; exports draw at their own.
#: What the window opens on: the first this many seconds of the recording.
OPENING_S = 60.0
PREVIEW_SIZE = (1280, 720)
#: The range fields, the look fields and the trail fields, in the order the form shows them.
RANGE_FIELDS = ("start_s", "end_s")
LOOK_FIELDS = ("location", "trail_s", "ghost_s", "ethogram_half_s", "location_window_s")
EXPORT_FIELDS = ("speed", "width", "workers")


class TreeCut(QWidget):
    """The clustering run's tree as a dendrogram with a cut you drag: every branch crossing the
    line is a numbered cluster, numbered by size, the coloured ones in their colour.

    Heights are ``rv.display_heights`` — raised to their children's, so the picture and the cut
    are the same monotone tree (``ClusterTree`` heights are monotone only within a transition).
    """

    #: The cut moved; emitted when the drag ends (a drag is many heights, a cut is one).
    cut_changed = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(QSize(lines(16), lines(9)))
        self.setMouseTracking(False)
        self._tree = None
        self._shown = None
        self._x = None
        self._cut = 0.0
        self._nodes: list[int] = []
        self._coloured: dict[int, int] = {}  # node -> colour code
        self._top = 1.0
        self._message = "Loading the tree…"

    # -- data ------------------------------------------------------------------------------- #
    def set_tree(self, children: np.ndarray, height: np.ndarray, n_leaves: int) -> None:
        self._tree = (np.asarray(children), np.asarray(height), int(n_leaves))
        self._shown = rv.display_heights(*self._tree)
        self._top = float(self._shown.max()) or 1.0
        self._x = self._leaf_positions()
        self._message = ""
        self.update()

    def set_message(self, text: str) -> None:
        self._message = text
        self.update()

    def set_cut(
        self, height: float, nodes: list[int], coloured: dict[int, int]
    ) -> None:
        self._cut, self._nodes, self._coloured = (
            float(height),
            list(nodes),
            dict(coloured),
        )
        self.update()

    def _leaf_positions(self) -> np.ndarray:
        children, _h, n = self._tree
        x = np.zeros(2 * n - 1)
        order, stack = [], [2 * n - 2]
        while stack:
            node = stack.pop()
            if node < n:
                order.append(node)
            else:
                a, b = children[node - n]
                stack.extend((int(b), int(a)))
        x[np.asarray(order, dtype=int)] = np.arange(len(order))
        for m in range(len(children)):
            a, b = children[m]
            x[n + m] = (x[int(a)] + x[int(b)]) / 2
        return x

    # -- geometry ----------------------------------------------------------------------------- #
    def _margin(self) -> tuple[float, float, float, float]:
        return lines(1), lines(1.5), lines(1), lines(0.5)  # left, top, right, bottom

    def _to_px(self, x: float, h: float) -> QPointF:
        left, top, right, bottom = self._margin()
        n = max(self._tree[2] - 1, 1)
        w, hh = self.width() - left - right, self.height() - top - bottom
        return QPointF(left + x / n * w, top + (1 - h / self._top) * hh)

    def _height_at(self, y: float) -> float:
        _left, top, _right, bottom = self._margin()
        hh = self.height() - top - bottom
        return float(np.clip((1 - (y - top) / max(hh, 1)) * self._top, 0.0, self._top))

    # -- painting ------------------------------------------------------------------------------ #
    def paintEvent(self, _event) -> None:  # noqa: N802 — Qt's name
        theme = plot_theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._tree is None:
            p.setPen(plot_theme.qcolor(theme.text_muted))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return
        children, _h, n = self._tree
        colour_of_node = self._node_colours()
        for m in range(len(children)):
            node = n + m
            a, b = (int(v) for v in children[m])
            colour = colour_of_node.get(node, theme.neutral)
            p.setPen(plot_theme.pen(colour))
            top = self._to_px(self._x[node], self._shown[node])
            for child in (a, b):
                lo = self._to_px(self._x[child], self._shown[child])
                p.drawLine(QPointF(lo.x(), lo.y()), QPointF(lo.x(), top.y()))
            p.drawLine(
                QPointF(self._to_px(self._x[a], 0).x(), top.y()),
                QPointF(self._to_px(self._x[b], 0).x(), top.y()),
            )
        y = self._to_px(0, self._cut).y()
        p.setPen(plot_theme.pen(theme.text, dashed=True))
        p.drawLine(QPointF(0, y), QPointF(self.width(), y))
        p.setPen(plot_theme.qcolor(theme.text))
        for i, node in enumerate(self._nodes):
            if len(self._nodes) > 40 and node not in self._coloured:
                continue
            pt = self._to_px(self._x[node], self._cut)
            p.drawText(QPointF(pt.x() - 4, y - 4), str(i + 1))

    def _node_colours(self) -> dict[int, str]:
        """Every node under a coloured cut node, in that node's colour."""
        if self._tree is None or not self._coloured:
            return {}
        children, _h, n = self._tree
        out = {}
        for node, code in self._coloured.items():
            colour = rv.colour_of(
                "light" if plot_theme.current().name != "dark" else "dark", code
            )
            stack = [node]
            while stack:
                cur = stack.pop()
                out[cur] = colour
                if cur >= n:
                    stack.extend(int(v) for v in children[cur - n])
        return out

    # -- the drag -------------------------------------------------------------------------------- #
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._tree is not None:
            self._cut = self._height_at(event.position().y())
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._tree is not None and event.buttons():
            self._cut = self._height_at(event.position().y())
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._tree is not None:
            self._cut = self._height_at(event.position().y())
            self.cut_changed.emit(self._cut)


class RecordingVisualizationWindow(Dialog):
    """Visualize one recording of clustering run ``run``: ``recording_ids`` are the run's own."""

    def __init__(
        self, experiment, run, recording_ids: list[str], *, parent=None
    ) -> None:
        super().__init__(parent)
        self._experiment, self._run = experiment, run
        self._recording_ids = list(recording_ids)
        self._viz = None
        self._generation = 0
        self._tree_nodes: list[int] = []
        self._pending_cut: tuple | None = None
        self.setWindowTitle(
            f"Recording Embedding and Location Visualization · {run.id}"
        )

        row = QHBoxLayout()
        self.body().addLayout(row, 1)
        controls = QVBoxLayout()
        row.addLayout(controls, 0)

        # recording + range
        controls.addWidget(_heading("Recording"))
        self._recording = QComboBox()
        for rid in self._recording_ids:
            self._recording.addItem(_recording_name(experiment, rid), rid)
        self._recording.currentIndexChanged.connect(self._reload)
        controls.addWidget(self._recording)
        range_host = QWidget()
        self._range_form = ConfigForm(range_host, RecordingVisualizationConfig)
        self._range: dict[str, QWidget] = {}
        for name in RANGE_FIELDS:
            widget = any_field_widget(RecordingVisualizationConfig, name)
            self._range[name] = widget
            self._range_form.addRow(
                f"{field_meta(RecordingVisualizationConfig, name)['label']}:", widget
            )
        self._range_form.finish()
        # Opens on the first minute: a whole recording's location is minutes of mask decoding,
        # and nobody asked for it yet. "Full recording" is one click.
        set_any_field_value(
            RecordingVisualizationConfig, "start_s", self._range["start_s"], 0.0
        )
        set_any_field_value(
            RecordingVisualizationConfig, "end_s", self._range["end_s"], OPENING_S
        )
        controls.addWidget(range_host)
        range_buttons = QHBoxLayout()
        self._apply_range = QPushButton("Show this range")
        self._apply_range.clicked.connect(self._reload)
        self._full = QPushButton("Full recording")
        self._full.clicked.connect(self._show_full)
        range_buttons.addWidget(self._apply_range)
        range_buttons.addWidget(self._full)
        range_buttons.addStretch(1)
        controls.addLayout(range_buttons)

        # clusters
        controls.addWidget(_heading("Clusters"))
        self._named = QRadioButton("Named clusters only")
        self._cut = QRadioButton("Numbered, from a cut on the tree")
        self._mode = QButtonGroup(self)
        self._mode.addButton(self._named)
        self._mode.addButton(self._cut)
        self._named.setChecked(True)
        self._named.toggled.connect(self._clusters_changed)
        controls.addWidget(self._named)
        controls.addWidget(self._cut)
        self._tree = TreeCut()
        self._tree.cut_changed.connect(self._cut_moved)
        controls.addWidget(self._tree)
        self._numbers = QListWidget()
        self._numbers.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._numbers.itemChanged.connect(self._ticks_changed)
        self._numbers.setMinimumHeight(lines(6))
        controls.addWidget(self._numbers)
        self._cluster_note = QLabel(
            f"Tick up to {MAX_COLOURED} to colour; the rest are grey."
        )
        self._cluster_note.setWordWrap(True)
        controls.addWidget(self._cluster_note)

        # look
        controls.addWidget(_heading("Look"))
        self._dark = QCheckBox("Black background")
        self._light = QCheckBox("White background")
        self._dark.setChecked(True)
        for box in (self._dark, self._light):
            box.toggled.connect(self._redraw)
        backgrounds = QHBoxLayout()
        backgrounds.addWidget(self._dark)
        backgrounds.addWidget(self._light)
        backgrounds.addStretch(1)
        controls.addLayout(backgrounds)
        look_host = QWidget()
        self._look_form = ConfigForm(look_host, RecordingVisualizationConfig)
        self._look: dict[str, QWidget] = {}
        for name in LOOK_FIELDS:
            widget = any_field_widget(RecordingVisualizationConfig, name)
            self._look[name] = widget
            self._look_form.addRow(
                f"{field_meta(RecordingVisualizationConfig, name)['label']}:", widget
            )
            signal = getattr(widget, "toggled", None) or getattr(widget, "valueChanged")
            signal.connect(
                self._reload if name == "location" else self._clusters_changed
            )
        self._look_form.finish()
        controls.addWidget(look_host)

        # saved
        controls.addWidget(_heading("Saved visualizations of this run"))
        self._saved = QListWidget()
        self._saved.setMinimumHeight(lines(4))
        self._saved.itemDoubleClicked.connect(self._open_saved)
        controls.addWidget(self._saved)
        saved_buttons = QHBoxLayout()
        self._open_btn = QPushButton("Open")
        self._open_btn.clicked.connect(
            lambda: self._open_saved(self._saved.currentItem())
        )
        self._delete_btn = QPushButton("Delete…")
        self._delete_btn.clicked.connect(self._delete_saved)
        saved_buttons.addWidget(self._open_btn)
        saved_buttons.addWidget(self._delete_btn)
        saved_buttons.addStretch(1)
        controls.addLayout(saved_buttons)
        controls.addStretch(1)

        # the picture
        self._view = ImageView(zoomable=True)
        self._view.setMinimumSize(QSize(lines(36), lines(20)))
        self._view.clear_frame("Loading the recording…")
        self._scrubber = FrameScrubber(playable=True)
        self._scrubber.set_decoder(self._compute)
        self._scrubber.rendered.connect(self._display)
        self._block = PictureWithControls(self._view, self._scrubber)
        picture = QVBoxLayout()
        picture.addWidget(self._block, 1)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        picture.addWidget(self._status)
        row.addLayout(picture, 1)

        actions = QHBoxLayout()
        self._save_btn = QPushButton("Save visualization")
        self._save_btn.clicked.connect(self._save)
        self._export_btn = QPushButton("Export…")
        self._export_btn.clicked.connect(self._export)
        actions.addWidget(self._save_btn)
        actions.addWidget(self._export_btn)
        actions.addStretch(1)
        self.body().addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)
        self.finished.connect(self._shut_down)
        self._sync()
        self._refresh_saved()
        self._reload()

    # -- the recipe -------------------------------------------------------------------------- #
    def recipe(self) -> RecordingVisualizationConfig:
        """The controls as a recipe.

        Raises:
            ValueError: The controls do not make a valid recipe (e.g. a range with one end).
        """
        values = {
            name: any_field_value(RecordingVisualizationConfig, name, widget)
            for name, widget in {**self._range, **self._look}.items()
        }
        values["recording_id"] = self._recording.currentData()
        if self._cut.isChecked():
            values["clusters"] = "cut"
            values["cut_height"] = self._cut_height()
            values["cut_included"] = self._ticked()
        return RecordingVisualizationConfig(**values)

    def themes(self) -> list[str]:
        out = [
            t
            for t, box in (("dark", self._dark), ("light", self._light))
            if box.isChecked()
        ]
        return out or ["dark"]

    def _cut_height(self) -> float:
        return float(self._tree._cut)

    def _ticked(self) -> list[int]:
        out = []
        for i in range(self._numbers.count()):
            item = self._numbers.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return out

    def set_recipe(self, recipe: RecordingVisualizationConfig) -> None:
        """Put a saved recipe in the controls and show it."""
        index = self._recording.findData(recipe.recording_id)
        if index >= 0:
            self._recording.blockSignals(True)
            self._recording.setCurrentIndex(index)
            self._recording.blockSignals(False)
        for name, widget in {**self._range, **self._look}.items():
            widget.blockSignals(True)
            set_any_field_value(
                RecordingVisualizationConfig, name, widget, getattr(recipe, name)
            )
            widget.blockSignals(False)
        self._cut.setChecked(recipe.clusters == "cut")
        self._named.setChecked(recipe.clusters == "named")
        self._pending_cut = (recipe.cut_height, list(recipe.cut_included))
        self._reload()

    # -- loading --------------------------------------------------------------------------- #
    def _reload(self, *_args) -> None:
        """Read the recording's data off the UI thread, then show its first frame."""
        self._generation += 1
        generation = self._generation
        self._scrubber.pause()
        self._viz = None
        self._sync()
        try:
            recipe = self.recipe()
        except ValueError as exc:
            self._view.clear_frame(str(exc))
            return
        run = self._run

        def work(emit):
            tree = run.hierarchy()
            return run.visualization(recipe, progress=emit), tree

        self._view.clear_frame("Loading the recording…")
        run_progress(
            self._block,
            work,
            lambda result: self._on_loaded(generation, result),
            label="Loading the recording…",
            on_error=self._failed,
            supersedes="recording visualization load",
        )

    def _on_loaded(self, generation: int, result) -> None:
        if generation != self._generation:
            return
        viz, tree = result
        self._viz = viz
        self._tree.set_tree(tree.children, tree.height, tree.n_leaves)
        self._fill_numbers(tree)
        d = viz.data
        self._scrubber.set_total(int(d.n_frames))
        self._scrubber.set_playback_rate(d.fps)
        self._scrubber.set_timestamps(np.asarray(d.t_ns[d.first : d.last], np.int64))
        self._status.setText(d.pose_note)
        self._sync()
        self._scrubber.set_frame(0)

    def _failed(self, exc: BaseException) -> None:
        self._view.clear_frame(str(exc))
        self._tree.set_message(str(exc))

    def _shut_down(self, *_args) -> None:
        self._generation += 1
        self._scrubber.pause()

    def _show_full(self) -> None:
        for name, widget in self._range.items():
            set_any_field_value(RecordingVisualizationConfig, name, widget, None)
        self._reload()

    # -- clusters --------------------------------------------------------------------------- #
    def _fill_numbers(self, tree) -> None:
        pending = self._pending_cut
        if pending and pending[0] is not None:
            height, included = pending
        elif self._tree_nodes:
            height, included = self._cut_height(), self._ticked()
        else:
            shown = np.sort(
                rv.display_heights(tree.children, tree.height, tree.n_leaves)[
                    tree.n_leaves :
                ]
            )[::-1]
            k = min(MAX_COLOURED, len(shown))
            height = (
                float((shown[k - 2] + shown[k - 1]) / 2)
                if k >= 2
                else float(shown[0] if len(shown) else 0)
            )
            included = list(range(1, MAX_COLOURED + 1))
        self._pending_cut = None
        self._set_cut(tree, height, included)

    def _set_cut(self, tree, height: float, included: list[int]) -> None:
        nodes = rv.cut_nodes(tree.children, tree.height, tree.n_leaves, height)
        rows = np.asarray(tree.n_rows, dtype=float)
        total = max(rows[2 * tree.n_leaves - 2], 1.0)
        self._tree_nodes = nodes
        self._numbers.blockSignals(True)
        self._numbers.clear()
        included = [n for n in included if 1 <= n <= len(nodes)][:MAX_COLOURED]
        for i, node in enumerate(nodes[:60]):
            number = i + 1
            item = QListWidgetItem(
                f"{number}    {rows[node] / total:.0%} of fitted frames"
            )
            item.setData(Qt.ItemDataRole.UserRole, number)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            on = number in included
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            code = included.index(number) if on else rv.GREY
            item.setData(
                Qt.ItemDataRole.DecorationRole,
                QColor(rv.colour_of(_screen_theme(), code)),
            )
            self._numbers.addItem(item)
        self._numbers.blockSignals(False)
        coloured = {nodes[n - 1]: i for i, n in enumerate(included)}
        self._tree.set_cut(height, nodes, coloured)
        more = f" (the first 60 of {len(nodes)} listed)" if len(nodes) > 60 else ""
        self._cluster_note.setText(
            f"{len(nodes)} clusters at this cut{more}. Tick up to {MAX_COLOURED} to colour; "
            "the rest are grey, as other."
        )

    def _cut_moved(self, height: float) -> None:
        tree = self._run.hierarchy()
        self._set_cut(tree, height, self._ticked())
        if not self._cut.isChecked():
            self._cut.setChecked(True)
        else:
            self._clusters_changed()

    def _ticks_changed(self, item: QListWidgetItem) -> None:
        if (
            item.checkState() == Qt.CheckState.Checked
            and len(self._ticked()) > MAX_COLOURED
        ):
            self._numbers.blockSignals(True)
            item.setCheckState(Qt.CheckState.Unchecked)
            self._numbers.blockSignals(False)
            self._cluster_note.setText(
                f"At most {MAX_COLOURED} clusters can be coloured; untick one first."
            )
            return
        self._set_cut(self._run.hierarchy(), self._cut_height(), self._ticked())
        self._clusters_changed()

    def _clusters_changed(self, *_args) -> None:
        """Re-colour (and re-read the trail settings) without reloading the recording."""
        if self._viz is None:
            return
        try:
            recipe = self.recipe()
        except ValueError as exc:
            self._status.setText(str(exc))
            return
        generation, viz, run = self._generation, self._viz, self._run

        def recolour():
            from detail.api.recording_visualization import (
                RecordingVisualization,
                recode,
            )

            return RecordingVisualization(recode(viz.data, run, recipe))

        run_busy(
            self._block,
            recolour,
            lambda new: self._on_recoded(generation, new),
            label="Colouring…",
            on_error=self._failed,
            supersedes="recording visualization recolour",
        )

    def _on_recoded(self, generation: int, viz) -> None:
        if generation != self._generation:
            return
        self._viz = viz
        self._redraw()

    # -- drawing ----------------------------------------------------------------------------- #
    def _compute(self, step: int):
        viz = self._viz
        if viz is None:
            return None
        generation, theme = self._generation, self.themes()[0]
        fno = viz.data.first + int(step)
        picture = frame_source.decode(lambda: viz.frame(fno, theme, PREVIEW_SIZE))
        return generation, picture

    def _display(self, _step: int, result) -> None:
        if result is None:
            return
        generation, (picture, reason) = result
        if generation != self._generation:
            return
        if reason is not None:
            self._view.clear_frame(frame_source.unavailable_text(reason))
            return
        self._view.set_frame(_to_pixmap(picture))
        self._block.set_aspect(self._view.pixmap_size())

    def _redraw(self, *_args) -> None:
        if self._viz is not None:
            self._scrubber.set_frame(self._scrubber.frame())

    def _sync(self) -> None:
        ready = self._viz is not None
        for button in (self._save_btn, self._export_btn):
            set_available(button, ready, "Opens once the recording is loaded.")
        has = self._saved.count() > 0
        for button in (self._open_btn, self._delete_btn):
            set_available(button, has, "No visualization of this run is saved yet.")

    # -- saving ------------------------------------------------------------------------------- #
    def _refresh_saved(self) -> None:
        self._saved.clear()
        for recipe in self._run.visualizations():
            item = QListWidgetItem(_describe(self._experiment, recipe))
            item.setData(Qt.ItemDataRole.UserRole, recipe["id"])
            self._saved.addItem(item)
        self._sync()

    def _save(self) -> None:
        try:
            viz_id = self._run.save_visualization(self.recipe())
        except ValueError as exc:
            self._status.setText(str(exc))
            return
        self._status.setText(f"Saved as {viz_id}.")
        self._refresh_saved()

    def _open_saved(self, item) -> None:
        if item is None:
            return
        from detail.api.recording_visualization import recipe_of

        viz_id = item.data(Qt.ItemDataRole.UserRole)
        saved = {r["id"]: r for r in self._run.visualizations()}
        self.set_recipe(recipe_of(saved[viz_id]))

    def _delete_saved(self) -> None:
        item = self._saved.currentItem()
        if item is None:
            return
        viz_id = item.data(Qt.ItemDataRole.UserRole)
        answer = QMessageBox.question(
            self,
            "Delete visualization",
            f"Delete the saved visualization {viz_id} ({item.text()})? It cannot be restored.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._run.delete_visualization(viz_id)
            self._refresh_saved()

    def _export(self) -> None:
        dialog = ExportVisualizationDialog(
            self._experiment,
            current=(self._run, self.recipe()) if self._viz is not None else None,
            themes=self.themes(),
            parent=self,
        )
        dialog.exec()


class ExportVisualizationDialog(Dialog):
    """Write one visualization, or a montage of saved ones from any experiment, side by side on
    one clock — with the time and size measured before anything is written."""

    def __init__(self, experiment, *, current=None, themes=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export visualization")
        self._experiment = experiment
        self._current = current
        #: The ticked visualizations' keys, in the order ticked: the tiles' order.
        self._order: list[str] = []
        layout = self.body()

        self._one = QRadioButton("This visualization")
        self._montage = QRadioButton(
            "Montage of saved visualizations, side by side on one clock"
        )
        group = QButtonGroup(self)
        group.addButton(self._one)
        group.addButton(self._montage)
        (self._one if current is not None else self._montage).setChecked(True)
        set_available(
            self._one,
            current is not None,
            "Open a recording in the visualization window first.",
        )
        self._one.toggled.connect(self._changed)
        layout.addWidget(self._one)
        layout.addWidget(self._montage)

        self._list = QListWidget()
        self._list.setMinimumHeight(lines(6))
        self._list.itemChanged.connect(self._ticked)
        layout.addWidget(self._list)
        list_buttons = QHBoxLayout()
        add = QPushButton("Add from another experiment…")
        add.clicked.connect(self._add_experiment)
        list_buttons.addWidget(add)
        list_buttons.addStretch(1)
        layout.addLayout(list_buttons)
        self._layout_note = QLabel("")
        self._layout_note.setWordWrap(True)
        layout.addWidget(self._layout_note)

        backgrounds = QHBoxLayout()
        self._dark = QCheckBox("Black background")
        self._light = QCheckBox("White background")
        for theme, box in (("dark", self._dark), ("light", self._light)):
            box.setChecked(theme in (themes or ["dark"]))
            box.toggled.connect(self._changed)
            backgrounds.addWidget(box)
        backgrounds.addStretch(1)
        layout.addLayout(backgrounds)
        form_host = QWidget()
        form = ConfigForm(form_host, RecordingVisualizationExport)
        self._fields: dict[str, QWidget] = {}
        for name in EXPORT_FIELDS:
            widget = any_field_widget(RecordingVisualizationExport, name)
            self._fields[name] = widget
            form.addRow(
                f"{field_meta(RecordingVisualizationExport, name)['label']}:", widget
            )
            signal = getattr(widget, "currentIndexChanged", None) or getattr(
                widget, "valueChanged"
            )
            signal.connect(self._changed)
        form.finish()
        layout.addWidget(form_host)

        path_row = QHBoxLayout()
        self._path = QLineEdit(str(Path.home() / "visualization.mp4"))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row.addWidget(self._path, 1)
        path_row.addWidget(browse)
        layout.addLayout(path_row)
        self._estimate = QLabel("Measuring…")
        self._estimate.setWordWrap(True)
        layout.addWidget(self._estimate)

        buttons = QDialogButtonBox()
        self._go = buttons.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._write)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)
        self._add_from(experiment)
        self._changed()

    # -- the list ----------------------------------------------------------------------------- #
    def _add_from(self, experiment) -> None:
        self._list.blockSignals(True)
        for run_id in _temporal_runs(experiment):
            run = experiment.temporal_clustering_run(run_id)
            for recipe in run.visualizations():
                label = f"{Path(experiment.dir).name} · {run_id} · {_describe(experiment, recipe)}"
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setData(
                    Qt.ItemDataRole.UserRole, (str(experiment.dir), run_id, recipe)
                )
                self._list.addItem(item)
        self._list.blockSignals(False)

    def _add_experiment(self) -> None:
        from detail.api.experiment import Experiment

        folder = QFileDialog.getExistingDirectory(self, "Add saved visualizations from")
        if not folder:
            return
        try:
            other = Experiment.open(folder)
        except (FileNotFoundError, ValueError) as exc:
            self._estimate.setText(str(exc))
            return
        self._add_from(other)
        self._changed()

    @staticmethod
    def _key(source: tuple) -> str:
        exp_dir, run_id, recipe = source
        return f"{exp_dir}::{run_id}::{recipe.get('id')}"

    def _ticked(self, item: QListWidgetItem) -> None:
        """A tick adds its visualization as the next tile and an untick takes it out: the order
        is the ticking order, so a visualization is known by its key, never by its row.
        """
        key = self._key(item.data(Qt.ItemDataRole.UserRole))
        on = item.checkState() == Qt.CheckState.Checked
        if on and key not in self._order:
            self._order.append(key)
        elif not on and key in self._order:
            self._order.remove(key)
        self._changed()

    def chosen(self) -> list[tuple]:
        """The montage's ``(experiment_dir, run_id, recipe)``, in the order ticked."""
        by_key = {}
        for i in range(self._list.count()):
            source = self._list.item(i).data(Qt.ItemDataRole.UserRole)
            by_key[self._key(source)] = source
        return [by_key[k] for k in self._order if k in by_key]

    def export_settings(self) -> RecordingVisualizationExport:
        values = {
            n: any_field_value(RecordingVisualizationExport, n, w)
            for n, w in self._fields.items()
        }
        values["themes"] = [
            t
            for t, b in (("dark", self._dark), ("light", self._light))
            if b.isChecked()
        ] or ["dark"]
        return RecordingVisualizationExport(**values)

    # -- estimate ------------------------------------------------------------------------------- #
    def _changed(self, *_args) -> None:
        n = 1 if self._one.isChecked() else len(self.chosen())
        if n == 0:
            self._layout_note.setText(
                "Tick the saved visualizations to tile, in the order they should appear."
            )
            self._estimate.setText("")
            set_available(self._go, False, "Nothing is chosen to export.")
            return
        if n > rv.MAX_MONTAGE:
            self._layout_note.setText(
                f"A montage holds at most {rv.MAX_MONTAGE}; untick {n - rv.MAX_MONTAGE}."
            )
            set_available(self._go, False, f"A montage holds at most {rv.MAX_MONTAGE}.")
            return
        settings = self.export_settings()
        tile = rv.montage_tiles(n, settings.width, settings.height)[0]
        cols, rows = rv.montage_grid(n)
        self._layout_note.setText(
            f"{cols} x {rows} on {settings.width} x {settings.height}: each tile "
            f"{int(tile.w)} x {int(tile.h)}, laid out for that size. Tiles start together; one that "
            "ends first holds its last frame, dimmed."
        )
        set_available(self._go, True, "")
        self._estimate.setText("Measuring a few frames…")
        sources, experiment = self._sources(), self._experiment

        def measure():
            from detail.api.recording_visualization import estimate

            return estimate(_open_all(experiment, sources), settings)

        run_busy(
            self._estimate,
            measure,
            self._show_estimate,
            label="Measuring…",
            on_error=lambda exc: self._estimate.setText(f"Could not measure: {exc}"),
            supersedes="recording visualization estimate",
        )

    def _sources(self) -> list[tuple]:
        if self._one.isChecked() and self._current is not None:
            run, recipe = self._current
            return [(str(self._experiment.dir), run.id, recipe.model_dump())]
        return self.chosen()

    def _show_estimate(self, est: dict) -> None:
        minutes = est["seconds"] / 60
        size = est["bytes_per_file"] / 1e6
        files = est["files"]
        self._estimate.setText(
            f"About {minutes:.0f} min ({est['workers']} workers) · about {size:.0f} MB per file · "
            f"{files} file{'s' if files > 1 else ''} · {est['frames']:,} frames each. Measured on "
            "a few frames drawn and encoded now."
        )

    # -- writing ------------------------------------------------------------------------------- #
    def _browse(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export to", self._path.text(), "MP4 (*.mp4)"
        )
        if path:
            self._path.setText(path)

    def _write(self) -> None:
        settings, sources, experiment = (
            self.export_settings(),
            self._sources(),
            self._experiment,
        )
        path = self._path.text()

        def work(emit):
            from detail.api.recording_visualization import export_montage

            return export_montage(
                _open_all(experiment, sources, emit), path, settings, emit
            )

        run_progress(
            self,
            work,
            lambda written: self._estimate.setText(
                "Wrote " + ", ".join(str(p) for p in written)
            ),
            label="Exporting…",
            on_error=lambda exc: self._estimate.setText(str(exc)),
            on_cancel=lambda: self._estimate.setText("Export stopped."),
            eta=True,
        )


# --------------------------------------------------------------------------- helpers


def _heading(text: str) -> QLabel:
    label = QLabel(f"<b>{text}</b>")
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _screen_theme() -> str:
    return "dark" if plot_theme.current().name == "dark" else "light"


def _recording_name(experiment, rid: str) -> str:
    try:
        return experiment.get_recording(rid).name
    except (KeyError, FileNotFoundError, ValueError):
        return rid


def _describe(experiment, recipe: dict) -> str:
    """One line for a saved recipe: recording, range, clusters, location."""
    name = _recording_name(experiment, recipe.get("recording_id", ""))
    if recipe.get("start_s") is None:
        span = "full recording"
    else:
        span = f"{_mmss(recipe['start_s'])}–{_mmss(recipe['end_s'])}"
    clusters = (
        "named"
        if recipe.get("clusters") == "named"
        else f"cut, {len(recipe.get('cut_included') or [])} coloured"
    )
    loc = "location" if recipe.get("location") else "no location"
    return f"{recipe.get('id', 'new')} · {name} · {span} · {clusters} · {loc}"


def _mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def _temporal_runs(experiment) -> list[str]:
    """The experiment's temporal-clustering runs, which are where saved visualizations live. A
    listing: one unreadable record costs its row, never the list (`storage/dirscan.read_each`).
    """
    from detail.storage import analysis_runs, dirscan

    records = dirscan.read_each(
        analysis_runs.discover_analyses(experiment.dir),
        lambda run_id: analysis_runs.read_analysis_yaml(
            analysis_runs.analysis_dir(experiment.dir, run_id)
        ),
    )
    return [r["id"] for r in records if r.get("module_id") == "temporal_clustering"]


def _open_all(experiment, sources: list[tuple], progress=None) -> list:
    """Each ``(experiment_dir, run_id, recipe)`` loaded, opening other experiments as needed."""
    from detail.api.experiment import Experiment

    experiments = {str(experiment.dir): experiment}
    out = []
    for i, (exp_dir, run_id, recipe) in enumerate(sources):
        exp = experiments.get(exp_dir) or experiments.setdefault(
            exp_dir, Experiment.open(exp_dir)
        )
        run = exp.temporal_clustering_run(run_id)
        step = None
        if progress is not None:
            step = lambda stage, f, d, i=i: progress(
                f"loading {i + 1}/{len(sources)}", 0.0, stage
            )
        out.append(run.visualization(recipe, progress=step))
    return out


def open_visualization_window(
    experiment, run, recording_ids: list[str], *, parent=None
):
    """Open :class:`RecordingVisualizationWindow` beside INTERPRET, non-modal: the tree stays
    usable while a recording plays (``requirements._NON_MODAL_DIALOGS``)."""
    window = RecordingVisualizationWindow(experiment, run, recording_ids, parent=parent)
    window.setModal(False)
    window.show()
    return window
