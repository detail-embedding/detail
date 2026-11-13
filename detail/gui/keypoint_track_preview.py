"""*Preview tracking…* (§26 *Previewing a track method*): the keypoint tracking form's window for
watching a few settings side by side over a few seconds of a few recordings.

The work is ``Experiment.preview_keypoint_tracking`` off the UI thread (``run_progress``, which
stops it if the window closes), with the form's own launch config, so every recording is
validated and resolved as a launch would. The window offers the recordings the form's settings
cover, two of them ticked (spread across the list), each with a start (*middle* until set) and
one duration for all; and the two **variants**, each a few of the method's settings
(``PREVIEW_VARIANT_FIELDS``, the widgets read off the method's own model) changed from the
form's, started from ``PREVIEW_STARTING_VARIANTS``.

The result is **one tile per setting**, played together: detection alone, the current settings
and each variant, on the same frame of the chosen recording, two to a row. Each tile is named
in its corner the way the form names it — *Variant A: transition 1* — and the table's rows
carry the same names, so a tile, its column of settings and its row of figures read as one. The
tiles are composed into **one picture** (:func:`mosaic`) under one scrubber, held together by
``PictureWithControls`` the way every player here is, so the controls sit under the picture
they drive rather than at the foot of the pane. The view zooms the **tiles together** and
paints their names (``ImageView.set_tiles``): the wheel magnifies the same region of every
tile, a drag pans them all, and the names keep the screen's text size — zooming the composed
picture would show one tile and lose the ones it is compared with, and names drawn into the
picture shrank to 7px with it. The figures table under it describes each setting without labels
(``analysis.keypoint_track_diagnostics``), detection alone first — the floor each column is read
against — and says so: a lower move or a steadier skeleton is what smoothing buys whether or not
the points stay on the animal, so the clips decide. **Use selected settings** hands the selected
row's full method config back to the form. Nothing is written.

**Search the grid** runs every *Tune tracking* grid setting over the same windows instead of the
variants (temporal_decode only). The table lists them front first (``PreviewResult.ranked``),
the front marked, and says the front is a provisional rule not yet checked against labels. Three
tiles play, not 38: detection alone, the current settings and **the selected row**, which the
first front row starts as; selecting another row redraws the frame with that setting.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QRect
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.analysis.keypoint_track_diagnostics import (
    METRIC_LABELS,
    METRICS,
    SHARES,
)
from detail.configs.keypoint_track import (
    METHOD_CONFIGS,
    PREVIEW_DEFAULT_RECORDINGS,
    PREVIEW_MAX_RECORDINGS,
    PREVIEW_MAX_VARIANTS,
    PREVIEW_STARTING_VARIANTS,
    PREVIEW_VARIANT_FIELDS,
    TUNE_METHOD,
    KeypointTrackPreviewWindow,
)
from detail.core.visualize import bgr_to_rgb, draw_keypoints
from detail.gui import frame_source
from detail.gui._pydantic_form import (
    any_field_value,
    any_field_widget,
    field_meta,
    optional_spin,
    optional_value,
    set_any_field_value,
)
from detail.gui.busy import decode_overlay, run_progress
from detail.gui.dialog import Dialog
from detail.gui.image_view import ImageView, PictureWithControls
from detail.gui.run_preview import _to_pixmap
from detail.gui.scrubber import FrameScrubber
from detail.gui.sizing import set_available
from detail.gui.tables import (
    fit_table_height,
    give_slack_to,
    list_table,
    num_cell,
    refit_slack,
    stamp_row_id,
)
from detail.gui.tables import row_id as table_row_id
from detail.pipeline.keypoint_track_preview import (
    ALL,
    CURRENT,
    DETECTION_ONLY,
    variant_name,
)
from detail.pipeline.subject_frames import choose_recordings

_NONE = "(none)"
_MIDDLE = "middle"
#: Each tile's width in the composed picture: the view scales the whole picture to fit, so
#: this only bounds the work of a frame (four tiles of a 1280x720 frame at half size).
_TILE_W = 640
#: The picture's least height, so the tiles stay readable in a window opened at its content.
_PICTURE_MIN_H = 360
#: The dark rule between tiles, so where one setting's footage ends is never a guess.
_GAP = 4
_GAP_COLOUR = (30, 30, 30)
#: What the form calls the two fixed settings; a variant is named by its column's letter.
_FIXED_LABELS = {DETECTION_ONLY: "Detection only", CURRENT: "Current settings"}
_ZOOM_HINT = "Wheel zooms every tile at once · drag to pan · Fit to see all"
_CAVEAT = (
    "These figures describe each setting; they do not say which is right. Smoothing lowers "
    "the move, the jumps and the swaps whether or not the points stay on the animal, so read "
    "every row against detection alone (the first) and judge by the clips."
)
_FRONT_NOTE = (
    "★ marks the front: no other setting beats it on agreement, placed, jumps and swaps "
    "together. It is a provisional rule, not yet checked against labels; watch a front row's "
    "clip before you use it."
)
_FRONT_MARK = "★"


class KeypointTrackPreviewDialog(Dialog):
    """Preview tracking settings on short windows; ``chosen_config`` is what the user took."""

    def __init__(
        self,
        experiment,
        preview_launch_config: dict,
        preview_recordings: list[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        """``preview_launch_config`` is the track form's ``config()``; ``preview_recordings``
        the ``(id, name)`` pairs those settings cover, in the form's order."""
        launch_config, recordings = preview_launch_config, preview_recordings
        super().__init__(parent)
        self.setWindowTitle("Preview tracking")
        self._experiment = experiment
        self._config = dict(launch_config)
        self._method = launch_config["method"]
        self._model = METHOD_CONFIGS[self._method]
        self._current = self._model.model_validate(
            launch_config.get("config") or {}
        ).model_dump()
        self._names = dict(recordings)
        self._result = None
        self._frames: dict = {}
        self._shown = None  # the recording whose clips are on screen
        self._tiles: list = []  # the settings the picture shows, read off the UI thread
        self._letters: dict[str, str] = (
            {}
        )  # a variant setting's name -> its column's letter
        self.chosen_config: dict | None = None
        intro = QLabel(
            "Tracking runs on a few seconds of a few recordings, for your current settings "
            "and the variants below, and the results play side by side with detection "
            "alone. It takes seconds rather than a full run's minutes, and nothing is "
            "written. Search the grid tries every Tune tracking setting instead."
        )
        intro.setWordWrap(True)
        self.body().addWidget(intro)
        self.body().addLayout(self._build_windows(recordings))
        self.body().addLayout(self._build_variants())
        self._run = QPushButton("Run preview")
        self._run.clicked.connect(self.run_preview)
        self._search = QPushButton("Search the grid")
        self._search.clicked.connect(self.search_grid)
        runs = QHBoxLayout()
        runs.addWidget(self._run)
        runs.addWidget(self._search)
        runs.addStretch(1)
        self.body().addLayout(runs)
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self.body().addWidget(self._summary)
        self._results = self._build_results()
        self._results.setVisible(False)
        self.body().addWidget(self._results)
        # The spare height goes to a spacer until there are results to give it to, so the
        # controls above never spread into it (tests/gui/test_dialog_space.py).
        self.body().addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._use = buttons.addButton(
            "Use selected settings", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._use.clicked.connect(self._take)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)
        self._table.selectionModel().selectionChanged.connect(self._selection_changed)
        self._sync()

    # -- the windows --------------------------------------------------------------- #
    def _build_windows(self, recordings: list[tuple[str, str]]) -> QFormLayout:
        form = QFormLayout()
        ids = [rid for rid, _name in recordings]
        ticked = choose_recordings(ids, PREVIEW_DEFAULT_RECORDINGS) if ids else []
        self._slots: list[tuple[QComboBox, QWidget]] = []
        start_help = field_meta(KeypointTrackPreviewWindow, "start_s")["help"]
        for slot in range(PREVIEW_MAX_RECORDINGS):
            combo = QComboBox()
            combo.addItem(_NONE, None)
            for rid, name in recordings:
                combo.addItem(name, rid)
            if slot < len(ticked):
                combo.setCurrentIndex(combo.findData(ticked[slot]))
            combo.currentIndexChanged.connect(self._sync)
            start = optional_spin(KeypointTrackPreviewWindow, "start_s", _MIDDLE)
            start.setToolTip(start_help)
            row = QHBoxLayout()
            row.addWidget(combo, 1)
            row.addWidget(QLabel("Start (s):"))
            row.addWidget(start)
            form.addRow(f"Recording {slot + 1}:", row)
            self._slots.append((combo, start))
        self._duration = any_field_widget(KeypointTrackPreviewWindow, "duration_s")
        form.addRow("Duration (s):", self._duration)
        return form

    def windows(self) -> tuple[list[str], dict[str, float | None]]:
        """The chosen recordings in slot order (each once) and their starts."""
        chosen: list[str] = []
        starts: dict[str, float | None] = {}
        for combo, start in self._slots:
            rid = combo.currentData()
            if rid is None or rid in starts:
                continue
            chosen.append(rid)
            starts[rid] = optional_value(start)
        return chosen, starts

    # -- the variants ------------------------------------------------------------ #
    def _build_variants(self) -> QGridLayout:
        """A column per variant, a row per variant field the method has; the current value
        beside each, read-only."""
        grid = QGridLayout()
        self._fields = [
            n for n in PREVIEW_VARIANT_FIELDS if n in self._model.model_fields
        ]
        grid.addWidget(QLabel("Setting"), 0, 0)
        grid.addWidget(QLabel("Current"), 0, 1)
        starting = PREVIEW_STARTING_VARIANTS.get(self._method, ())
        self._variants: list[tuple[QCheckBox, dict[str, QWidget]]] = []
        for v in range(PREVIEW_MAX_VARIANTS):
            tick = QCheckBox(f"Try variant {chr(ord('A') + v)}")
            tick.setChecked(True)
            tick.toggled.connect(self._sync)
            grid.addWidget(tick, 0, 2 + v)
            widgets: dict[str, QWidget] = {}
            overrides = starting[v] if v < len(starting) else {}
            for r, name in enumerate(self._fields, start=1):
                widget = any_field_widget(self._model, name)
                set_any_field_value(
                    self._model,
                    name,
                    widget,
                    overrides.get(name, self._current[name]),
                )
                grid.addWidget(widget, r, 2 + v)
                widgets[name] = widget
            self._variants.append((tick, widgets))
        for r, name in enumerate(self._fields, start=1):
            meta = field_meta(self._model, name)
            label = QLabel(meta["label"] + ":")
            label.setToolTip(meta["help"])
            grid.addWidget(label, r, 0)
            grid.addWidget(QLabel(_shown(self._current[name])), r, 1)
        return grid

    def variants(self) -> list[dict]:
        """Each ticked variant's settings that differ from the current ones (an unchanged one
        is dropped: it would repeat the current column)."""
        return [changed for _letter, changed in self._lettered_variants()]

    def _lettered_variants(self) -> list[tuple[str, dict]]:
        """:meth:`variants`, each with the letter of the column it came from — a repeat keeps
        the first column's."""
        out: list[tuple[str, dict]] = []
        for column, (tick, widgets) in enumerate(self._variants):
            if not tick.isChecked():
                continue
            overrides = {
                name: any_field_value(self._model, name, widget)
                for name, widget in widgets.items()
            }
            changed = {k: v for k, v in overrides.items() if v != self._current[k]}
            if changed and all(changed != seen for _letter, seen in out):
                out.append((chr(ord("A") + column), changed))
        return out

    def label(self, name: str) -> str:
        """A setting as the window shows it: *Detection only*, *Current settings*, *Variant A:
        transition 1*; a grid setting by its own name."""
        if name in _FIXED_LABELS:
            return _FIXED_LABELS[name]
        letter = self._letters.get(name)
        return name if letter is None else f"Variant {letter}: {name}"

    # -- the results ------------------------------------------------------------- #
    def _build_results(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        self._show_combo = QComboBox()
        self._show_combo.currentIndexChanged.connect(self._show_recording)
        clips = _labelled("Clips of:", self._show_combo)
        clips.addWidget(QLabel(_ZOOM_HINT))
        layout.addLayout(clips)
        self._view = ImageView(zoomable=True)
        self._scrubber = FrameScrubber(playable=True, loop=True)
        self._scrubber.set_decoder(self._compute)
        self._scrubber.rendered.connect(self._display)
        self._decode_busy = decode_overlay(self._scrubber, self._view)
        self._player = PictureWithControls(self._view, self._scrubber)
        self._player.setMinimumHeight(_PICTURE_MIN_H)
        # The picture is what the window is for: it takes the height the rest does not need.
        layout.addWidget(self._player, 1)
        self._figures_combo = QComboBox()
        self._figures_combo.currentIndexChanged.connect(self._fill_table)
        layout.addLayout(_labelled("Figures for:", self._figures_combo))
        self._table_model = QStandardItemModel(0, 2 + len(METRICS))
        self._table_model.setHorizontalHeaderLabels(
            ["Setting", "Front"] + [METRIC_LABELS[m] for m in METRICS]
        )
        self._table = list_table(stretch=False)
        self._table.setModel(self._table_model)
        give_slack_to(self._table, 0)
        layout.addWidget(self._table)
        caveat = QLabel(_CAVEAT)
        caveat.setWordWrap(True)
        layout.addWidget(caveat)
        self._front_note = QLabel(_FRONT_NOTE)
        self._front_note.setWordWrap(True)
        layout.addWidget(self._front_note)
        return box

    # -- state ------------------------------------------------------------------- #
    def _selected(self):
        if self._result is None:
            return None
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        name = table_row_id(rows[0])
        return next((s for s in self._result.settings if s.name == name), None)

    def _sync(self, *_args) -> None:
        chosen, _starts = self.windows()
        set_available(
            self._run, bool(chosen), "Choose at least one recording to preview"
        )
        if self._method != TUNE_METHOD:
            reason = (
                f"The grid covers {TUNE_METHOD} settings; choose {TUNE_METHOD} on the "
                "form to search it"
            )
        elif not chosen:
            reason = "Choose at least one recording to search"
        else:
            reason = ""
        set_available(self._search, not reason, reason)
        picked = self._selected()
        if picked is None:
            reason = "Run the preview, then select a row"
        elif picked.overrides is None:
            reason = "Detection alone is not a tracking setting; select another row"
        else:
            reason = ""
        set_available(self._use, not reason, reason)

    # -- running ----------------------------------------------------------------- #
    def run_preview(self) -> None:
        self._start(grid=False)

    def search_grid(self) -> None:
        self._start(grid=True)

    def _start(self, *, grid: bool) -> None:
        chosen, starts = self.windows()
        if not chosen:
            return
        set_available(self._run, False, "Previewing…")
        set_available(self._search, False, "Previewing…")
        experiment, config = self._experiment, dict(self._config)
        duration = any_field_value(
            KeypointTrackPreviewWindow, "duration_s", self._duration
        )
        lettered = [] if grid else self._lettered_variants()
        variants = None if grid else [changed for _letter, changed in lettered]
        letters = {variant_name(changed): letter for letter, changed in lettered}

        def work(emit):  # run_progress below runs this off the UI thread
            result = experiment.preview_keypoint_tracking(  # off-thread
                config,
                chosen,
                starts=starts,
                duration_s=duration,
                variants=variants,
                grid=grid,
                progress=emit,
            )
            frames = {
                w.recording_id: _WindowFrames(experiment, w) for w in result.windows
            }
            return result, frames, letters

        run_progress(
            self,
            work,
            self._show,
            label="Searching the grid…" if grid else "Previewing tracking…",
            on_error=self._failed,
            supersedes="preview",
            on_cancel=self._cancelled,
        )

    def _failed(self, exc: BaseException) -> None:
        self._summary.setText(f"Could not preview: {exc}")
        self._sync()

    def _cancelled(self) -> None:
        self._summary.setText("Stopped.")
        self._sync()

    def _show(self, done) -> None:
        self._result, self._frames, self._letters = done
        result = self._result
        self._summary.setText(
            f"{result.method} over "
            + "; ".join(
                f"{self._names.get(w.recording_id, w.recording_id)} "
                f"{w.start_ns / 1e9:.1f}–{w.end_ns / 1e9:.1f} s"
                for w in result.windows
            )
            + f". The figures cover the tracked points ({', '.join(result.tracked)})."
        )
        self._show_combo.blockSignals(True)
        self._figures_combo.blockSignals(True)
        self._show_combo.clear()
        self._figures_combo.clear()
        self._figures_combo.addItem("All recordings", ALL)
        for w in result.windows:
            name = self._names.get(w.recording_id, w.recording_id)
            self._show_combo.addItem(name, w.recording_id)
            self._figures_combo.addItem(name, w.recording_id)
        self._show_combo.blockSignals(False)
        self._figures_combo.blockSignals(False)
        self._table.setColumnHidden(1, not result.grid)
        self._front_note.setVisible(result.grid)
        self._results.setVisible(True)
        # The results take the spare height from the spacer that held it (see __init__).
        body = self.body()
        body.setStretchFactor(self._results, 1)
        body.setStretch(body.count() - 1, 0)
        self._fill_table()
        if result.grid and result.front:
            # the clips start on the first front row (ranked: detection, current, front…)
            self._select(result.ranked()[2].name)
        self._retile()
        self._show_recording()
        self.content_grew()
        self._sync()

    def _fill_table(self, *_args) -> None:
        if self._result is None:
            return
        picked = self._selected()
        key = self._figures_combo.currentData() or ALL
        front = set(self._result.front)
        self._table_model.removeRows(0, self._table_model.rowCount())
        for setting in self._result.ranked():
            name = QStandardItem(self.label(setting.name))
            name.setEditable(False)
            stamp_row_id(name, setting.name)
            mark = QStandardItem(_FRONT_MARK if setting.name in front else "")
            mark.setEditable(False)
            figures = setting.figures.get(key, {})
            self._table_model.appendRow(
                [name, mark] + [_figure_cell(m, figures.get(m)) for m in METRICS]
            )
        self._table.resizeColumnsToContents()
        refit_slack(self._table)
        fit_table_height(self._table, fixed=True)
        if picked is not None:
            self._select(picked.name)
        self._sync()

    def _select(self, name: str) -> None:
        for row in range(self._table_model.rowCount()):
            if table_row_id(self._table_model.index(row, 0)) == name:
                self._table.selectRow(row)
                return

    def _selection_changed(self, *_args) -> None:
        self._sync()
        if self._result is not None and self._result.grid:
            self._retile()
            if self._shown is not None:
                self._scrubber.set_frame(self._scrubber.frame())

    def _retile(self) -> None:
        """The settings the picture shows: every one of a preview's; for a grid search,
        detection alone, the current settings and the selected row."""
        result = self._result
        if not result.grid:
            self._tiles = list(result.settings)
            return
        tiles = [result.setting(DETECTION_ONLY), result.setting(CURRENT)]
        picked = self._selected()
        if picked is not None and picked.name not in (DETECTION_ONLY, CURRENT):
            tiles.append(picked)
        self._tiles = tiles

    def _show_recording(self, *_args) -> None:
        rid = self._show_combo.currentData()
        if rid is None or rid not in self._frames:
            return
        self._shown = rid
        frames = self._frames[rid]
        self._view.fit()  # another recording's animal is elsewhere in the frame
        self._scrubber.pause()
        self._scrubber.set_total(frames.n_frames)
        self._scrubber.set_timestamps(frames.t_ns)
        self._scrubber.set_playback_rate(frames.fps)
        self._scrubber.set_frame(0)

    # -- frames ------------------------------------------------------------------ #
    def _compute(self, index: int):
        """Off the UI thread (the scrubber's decoder): the frame, once per setting with that
        setting's points drawn on each subject, composed into one picture, and where in it
        each tile's footage lies."""
        rid, result, settings = self._shown, self._result, self._tiles
        if rid is None or result is None or not settings:
            return None
        frames = self._frames[rid]
        frame_idx = frames.first_frame + int(index)
        rgb, reason = frame_source.decode(lambda: frames.read(frame_idx))
        if rgb is None:
            return reason
        h, w = rgb.shape[:2]
        if (
            w > _TILE_W
        ):  # drawn at tile size: the points are normalized, so any size works
            rgb = cv2.resize(
                rgb, (_TILE_W, round(h * _TILE_W / w)), interpolation=cv2.INTER_AREA
            )
        skeleton = result.skeleton
        tiles = []
        for setting in settings:
            tile = rgb
            for track in setting.tracks.get(rid, {}).values():
                at = int(np.searchsorted(track.frame_idx, frame_idx))
                if at < len(track.frame_idx) and track.frame_idx[at] == frame_idx:
                    tile = draw_keypoints(tile, track.points[at], skeleton)
            tiles.append(tile)
        names = [self._tile_label(s, grid=result.grid) for s in settings]
        return mosaic(tiles), tile_rects(len(tiles), *tiles[0].shape[:2]), names

    def _tile_label(self, setting, *, grid: bool) -> str:
        """A tile's name: the row's, and for the grid's third tile, that it is the row picked."""
        label = self.label(setting.name)
        if grid and setting.name not in _FIXED_LABELS:
            return f"Selected row: {label}"
        return label

    def _display(self, _index: int, payload) -> None:
        if isinstance(payload, str) or payload is None:
            text = (
                frame_source.unavailable_text(payload)
                if isinstance(payload, str)
                else ""
            )
            self._view.clear_frame(text)
            return
        picture, rects, names = payload
        self._view.set_tiles([QRect(*rect) for rect in rects], names)
        self._view.set_frame(_to_pixmap(picture))
        self._player.set_aspect(self._view.pixmap_size())

    def _take(self) -> None:
        picked = self._selected()
        if picked is None or picked.overrides is None:
            return
        self.chosen_config = self._result.config_of(picked.name)
        self.accept()


class _WindowFrames:
    """One window's video: built off the UI thread (it reads the part's timestamp map), read
    on the scrubber's worker through a cursor, so playing forward decodes each frame once.
    """

    def __init__(self, experiment, window) -> None:
        from detail.storage import experiment as experiment_store
        from detail.storage import recording as recording_store
        from detail.storage.video import FrameCursor

        rec_dir = experiment_store.recording_path(experiment.dir, window.recording_id)
        data = recording_store.read_recording(rec_dir)
        part = recording_store.video_files(data)[window.part_index]
        self._video = recording_store.resolve_part_video_path(rec_dir, part)
        self._timestamps = recording_store.load_active_timestamps_cached(
            rec_dir, data, window.part_index
        ).values
        self.first_frame = int(np.searchsorted(self._timestamps, window.start_ns))
        last = int(np.searchsorted(self._timestamps, window.end_ns))
        self.n_frames = max(last - self.first_frame, 0)
        self.t_ns = self._timestamps[self.first_frame : last]
        self.fps = float(data["video"]["fps_actual"])
        self._cursor = FrameCursor(self._video, timestamps=self._timestamps)

    def read(self, frame_idx: int) -> np.ndarray | None:
        """The frame as RGB, or None when this position has no picture."""
        from detail.storage.video import read_frame_at

        decoded = self._cursor.frame(frame_idx)
        if decoded is None:
            decoded = read_frame_at(self._video, frame_idx, timestamps=self._timestamps)
        return None if decoded is None else bgr_to_rgb(decoded[1])


def mosaic(tiles: list[np.ndarray]) -> np.ndarray:
    """``tiles`` (RGB, one size) two to a row, a dark rule between them: one picture, so one
    view and one transport show every setting. An odd last row is padded. The names are the
    view's to draw (``ImageView.set_tiles``), at a size the screen can read.
    """
    h, w = tiles[0].shape[:2]
    rects = tile_rects(len(tiles), h, w)
    width = max(x + w for x, _y, _w, _h in rects)
    height = max(y + h for _x, y, _w, _h in rects)
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = _GAP_COLOUR
    for tile, (x, y, _w, _h) in zip(tiles, rects):
        canvas[y : y + h, x : x + w] = tile
    return canvas


def tile_rects(n_tiles: int, h: int, w: int) -> list[tuple[int, int, int, int]]:
    """``(x, y, width, height)`` of each tile in :func:`mosaic`'s picture — what the view
    zooms together and names. An odd last row's empty cell is no tile."""
    columns = 1 if n_tiles == 1 else 2
    return [
        ((i % columns) * (w + _GAP), (i // columns) * (h + _GAP), w, h)
        for i in range(n_tiles)
    ]


def _labelled(label: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(QLabel(label))
    row.addWidget(widget, 1)
    return row


def _figure_cell(metric: str, value) -> QStandardItem:
    if value is None:
        cell = QStandardItem("–")
        cell.setEditable(False)
        return cell
    text = f"{value:.1%}" if metric in SHARES else f"{value:.2f}"
    return num_cell(value, text)


def _shown(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
