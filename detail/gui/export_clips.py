"""A clip grid's one **Export clips…** dialog (§22 *Clip montage*, *Export clips*).

Opened from **Export ▾ → Export clips…** on every clip grid — INTERPRET's montage, a cluster's
pop-out window, AUDIT's review grid (``montage_export.MontageExport``, which it reads live) —
with *Clips* set to the picks while pick mode is on. Everything visual the grid can write is
here — the montage file, one file per clip, or both; MP4 (in the codec chosen) and/or GIF, each
at its own width and frame rate; and as many looks as are ticked, each into its own subfolder.
The CSV list stays its own **Export ▾** item, because it is data rather than a picture. The
writing is ``api.montage_export``'s; this is the asking, the preview and the waiting. Every
clip is written widened by **Show ▾**'s context, labelled by its own span, as the quick
writers do.

**The rows** are the looks (``montage_export.VISUALS``) under one for **Show ▾**'s look as it
is now (``show_menu.layers_of``), whose name follows it while the dialog is open. Rows that draw
keypoints are greyed, with the reason, when none of the clips' embedding runs read a keypoint
run. One **Keypoint trail points** choice serves every row with a keypoint trail, starting from
**Show ▾**'s *Trail points*.

**The preview** plays the grid's first clip (#1 in its order) in the row whose **Preview** is
pressed, drawn exactly as its own file would be: at the MP4's width and rate, or the GIF's when
only GIF is ticked, and it says which. It is drawn by the scrubber's background decoder, never
on the UI thread.

**Nothing waits on the UI thread.** Opening reads which clips have keypoints and each clip's
crop size behind the scenes; every change re-estimates the size and time off the thread; the
export runs through ``busy.run_progress`` — one bar over every file, with Cancel, which keeps the
finished files and removes the one being written.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings, QSize, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidgetAction,
)

from detail.api import clip_montage as montage_api
from detail.api import montage_export as export_api
from detail.api.animation import (
    GIF,
    GIF_DEFAULT_FPS,
    GIF_DEFAULT_WIDTH,
    GIF_FPS_RANGE,
    GIF_WIDTH_RANGE,
    MP4,
    sample_offsets_ns,
)
from detail.gui import frame_source
from detail.gui.animation_export import safe_name
from detail.gui.busy import run_progress
from detail.gui.dialog import Dialog
from detail.gui.image_view import ImageView, PictureWithControls
from detail.gui.run_preview import _to_pixmap
from detail.gui.scrubber import FrameScrubber
from detail.gui.sizing import lines
from detail.gui.skeleton_points import SkeletonPointsPicker
from detail.gui.spin import FloatSpin, IntSpin
from detail.gui.video_codec import VideoCodecPicker, with_codec_note
from detail.gui.workers import BackgroundCall, StageStatusLine

#: Where the last export went, per experiment: the next one opens beside it.
FOLDER_KEY = "clip_montage/export_parent/"
#: The MP4 widths offered under full size, in pixels.
MP4_WIDTHS = (1920, 1280, 960, 640, 480, 320)
#: The widest an MP4's rate box goes.
MP4_FPS_MAX = 120.0
NO_KEYPOINTS = "These clips' embedding runs read no keypoint run, so there are no keypoints to draw."
CHECKING_KEYPOINTS = "Checking whether these clips have keypoints…"
_HINT = (
    "Click a row's Preview to see that visual on this clip. It plays at the export's own "
    "size and frame rate."
)


def default_parent() -> Path:
    """Where an export goes before one has been made from this experiment."""
    documents = Path.home() / "Documents"
    return (documents if documents.is_dir() else Path.home()) / "detail exports"


class MontageExportDialog(Dialog):
    """Choose looks, files, formats and a folder, preview them, and export (module docstring).

    ``exports`` is the grid's ``montage_export.MontageExport``: its clips in grid order, the
    picks, **Show ▾**'s look and context are read from it, live, and what an export wrote is
    said on the line over its grid.

    Raises:
        ValueError: the grid has no clips (**Export ▾** greys the entry with the reason).
    """

    def __init__(self, exports, *, selected_only: bool = False, parent=None) -> None:
        super().__init__(parent if parent is not None else exports.montage().window())
        self._window = exports
        members = exports.members()
        if not members:
            raise ValueError("there are no clips to export")
        self._rank1 = members[0]
        #: Every file and the preview draw on this one montage's decode contexts and cache;
        #: its threads stop when the dialog closes.
        self._root = montage_api.ClusterMontage(
            exports.experiment_dir(), self._wide(members)
        )
        self._native: dict = {}
        self._footage_fps: float | None = None
        self._has_keypoints: bool | None = None
        self._skeleton = exports.trail_skeleton()
        self._costs = export_api.DEFAULT_COSTS
        self._estimate_generation = 0
        self._preview_generation = 0
        self._preview_spec = None
        self._exporting = False
        self._mp4_fps_touched = False
        self.setWindowTitle(f"{exports.title_text()} · export clips")

        layout = self.body()
        top = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self._build_visuals())
        left.addWidget(self._build_what())
        left.addWidget(self._build_formats())
        left.addStretch(1)
        top.addLayout(left, 3)
        right = QVBoxLayout()
        right.addWidget(self._build_preview())
        right.addStretch(1)
        top.addLayout(right, 2)
        layout.addLayout(top, 1)
        layout.addLayout(self._build_folder())
        self._summary = QLabel("")
        self._summary.setObjectName("export_summary")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        self._status = QLabel("")
        self._status.setObjectName("export_status")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QDialogButtonBox()
        self._export = buttons.addButton(
            "Export", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._export.setObjectName("export_go")
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.export)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)

        self._rows[export_api.CURRENT][1].setChecked(True)
        if selected_only:
            self.set_selected_only(True)
        self.finished.connect(self._shut_down)
        self.refresh()
        self._load()

    def _wide(self, clips) -> list:
        """``clips`` widened by **Show ▾**'s context: what a tile plays and a file writes."""
        return montage_api.with_context(list(clips), self._window.context_s())

    def _wide_one(self, clip):
        return self._wide([clip])[0]

    # -- building ------------------------------------------------------------------ #
    def _build_visuals(self) -> QGroupBox:
        box = QGroupBox("Visuals: each ticked row is exported as its own set of files")
        grid = QGridLayout(box)
        self._preview_group = QButtonGroup(self)
        self._preview_group.setExclusive(True)
        self._rows: dict[str, tuple[QCheckBox, QPushButton]] = {}
        keys = [export_api.CURRENT, *export_api.VISUALS_BY_KEY]
        line = 0
        for key in keys:
            visual = export_api.VISUALS_BY_KEY.get(key)
            tick = QCheckBox(visual.label if visual else "Current preview settings")
            tick.setObjectName(f"export_visual_{key}")
            tick.setChecked(key == export_api.CURRENT)
            tick.toggled.connect(lambda _on: self.refresh())
            show = QPushButton("Preview")
            show.setObjectName(f"export_preview_{key}")
            show.setCheckable(True)
            show.toggled.connect(lambda on, k=key: self._on_preview_pressed(k, on))
            self._preview_group.addButton(show)
            grid.addWidget(tick, line, 0)
            grid.addWidget(show, line, 1)
            self._rows[key] = (tick, show)
            line += 1
            if key == export_api.CURRENT:
                font = QFont(tick.font())
                font.setBold(True)
                tick.setFont(font)
                tick.setToolTip(
                    "What the montage draws now, with its ticks. Its name follows the ticks "
                    "while this window is open."
                )
                rule = QFrame()
                rule.setFrameShape(QFrame.Shape.HLine)
                grid.addWidget(rule, line, 0, 1, 2)
                line += 1
        grid.setColumnStretch(0, 1)
        points = QHBoxLayout()
        points.addWidget(QLabel("Keypoint trail points:"))
        self._picker = SkeletonPointsPicker(wording="trail")
        self._picker.changed.connect(self.refresh)
        self._points = QPushButton("every point")
        self._points.setObjectName("export_trail_points")
        menu = QMenu(self._points)
        holder = QWidgetAction(menu)
        holder.setDefaultWidget(self._picker)
        menu.addAction(holder)
        self._points.setMenu(menu)
        points.addWidget(self._points)
        points.addStretch(1)
        grid.addLayout(points, line, 0, 1, 2)
        line += 1
        self._align = QCheckBox("Align to heading")
        self._align.setObjectName("export_align")
        self._align.setChecked(bool(self._window.layers().aligned))
        self._align.setToolTip(
            "Turn every tile of the looks listed under the current one so its animal faces "
            "up, by the heading the embedding uses; the current look follows the montage's "
            "own Align tick."
        )
        self._align.toggled.connect(lambda _on: self.refresh())
        grid.addWidget(self._align, line, 0, 1, 2)
        if self._skeleton is not None:
            self._picker.set_skeleton(*self._skeleton)
            self._picker.set_value(self._window.trail_names())
        return box

    def _build_what(self) -> QGroupBox:
        box = QGroupBox("What to write")
        column = QVBoxLayout(box)
        self._write_group = QButtonGroup(self)
        self._writes: dict[str, QRadioButton] = {}
        for key, text in export_api.WRITES.items():
            radio = QRadioButton(text)
            radio.setObjectName(f"export_write_{key}")
            radio.setChecked(key == export_api.DEFAULT_WRITE)
            radio.toggled.connect(lambda _on: self.refresh())
            self._write_group.addButton(radio)
            column.addWidget(radio)
            self._writes[key] = radio
        row = QHBoxLayout()
        row.addWidget(QLabel("Clips:"))
        self._clips = QComboBox()
        self._clips.setObjectName("export_clips")
        self._clips.addItems(["All in the grid", "Picked only"])
        self._clips.setToolTip(
            "Every clip the grid shows, in its order — or only the clips picked in pick mode "
            "(Export ▾ → Pick clips to export…), in pick order."
        )
        self._clips.currentIndexChanged.connect(lambda _i: self.refresh())
        row.addWidget(self._clips)
        row.addStretch(1)
        column.addLayout(row)
        return box

    def _build_formats(self) -> QGroupBox:
        box = QGroupBox("Formats")
        grid = QGridLayout(box)
        self._mp4 = QCheckBox("MP4")
        self._mp4.setObjectName("export_mp4")
        self._mp4.setChecked(True)
        self._mp4.setToolTip("A video to keep or step through frame by frame.")
        self._mp4_width = QComboBox()
        self._mp4_width.setObjectName("export_mp4_width")
        self._mp4_width.addItem("full size", None)
        for width in MP4_WIDTHS:
            self._mp4_width.addItem(f"{width} px", width)
        self._mp4_width.setToolTip(
            "Full size draws each clip at its crop's own size in the recording, so nothing "
            "is shrunk. A montage file's grid is kept within "
            f"{montage_api.EXPORT_MAX_WIDTH} px wide. A width is the file's width at most."
        )
        self._mp4_fps = FloatSpin()
        self._mp4_fps.setObjectName("export_mp4_fps")
        self._mp4_fps.setRange(GIF_FPS_RANGE[0], MP4_FPS_MAX)
        self._mp4_fps.setValue(montage_api.MONTAGE_MAX_FPS)
        self._mp4_fps.setToolTip(
            "Pictures a second; the recording's own rate to begin with. Any rate keeps real "
            "time: a lower one skips frames rather than playing slower."
        )
        self._mp4_fps.valueChanged.connect(self._on_mp4_fps)
        self._mp4_codec = VideoCodecPicker()
        self._mp4_codec.changed.connect(lambda _codec: self.refresh())
        self._gif = QCheckBox("GIF")
        self._gif.setObjectName("export_gif")
        self._gif.setChecked(True)
        self._gif.setToolTip("An animation to drop into a slide or a message.")
        self._gif_width = IntSpin()
        self._gif_width.setObjectName("export_gif_width")
        self._gif_width.setRange(*GIF_WIDTH_RANGE)
        self._gif_width.setSingleStep(32)
        self._gif_width.setSuffix(" px")
        self._gif_width.setValue(GIF_DEFAULT_WIDTH)
        self._gif_width.setToolTip(
            "The GIF's width; its height follows. A narrower GIF is a smaller file."
        )
        self._gif_fps = FloatSpin()
        self._gif_fps.setObjectName("export_gif_fps")
        self._gif_fps.setRange(*GIF_FPS_RANGE)
        self._gif_fps.setValue(GIF_DEFAULT_FPS)
        self._gif_fps.setToolTip(
            "Pictures a second. A GIF keeps real time at any rate: a lower one skips frames "
            "and makes a smaller file."
        )
        # The codec sits on the row under MP4's, so it reads as the MP4's and not the GIF's.
        for row, (tick, width, fps) in (
            (0, (self._mp4, self._mp4_width, self._mp4_fps)),
            (2, (self._gif, self._gif_width, self._gif_fps)),
        ):
            grid.addWidget(tick, row, 0)
            grid.addWidget(QLabel("width"), row, 1)
            grid.addWidget(width, row, 2)
            grid.addWidget(QLabel("fps"), row, 3)
            grid.addWidget(fps, row, 4)
            tick.toggled.connect(lambda _on: self.refresh())
        grid.addWidget(QLabel("codec"), 1, 1)
        grid.addWidget(self._mp4_codec, 1, 2, 1, 3)
        self._mp4_width.currentIndexChanged.connect(lambda _i: self.refresh())
        self._gif_width.valueChanged.connect(lambda _v: self.refresh())
        self._gif_fps.valueChanged.connect(lambda _v: self.refresh())
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(4, 1)
        self._tips = {
            widget: widget.toolTip()
            for widget in (
                self._mp4_width,
                self._mp4_fps,
                self._mp4_codec,
                self._gif_width,
                self._gif_fps,
            )
        }
        return box

    def _build_preview(self) -> QGroupBox:
        box = QGroupBox("Preview: #1, the grid's first clip")
        column = QVBoxLayout(box)
        self._view = ImageView(zoomable=False)
        self._view.setObjectName("export_preview_picture")
        self._view.setMinimumSize(QSize(lines(24), lines(24)))
        self._view.clear_frame("Preparing the preview…")
        self._scrubber = FrameScrubber(playable=True, loop=True)
        self._scrubber.set_decoder(self._compute)
        self._scrubber.rendered.connect(self._display)
        self._loop = QCheckBox("Loop")
        self._loop.setObjectName("export_preview_loop")
        self._loop.setChecked(True)
        self._loop.setToolTip("Start the clip again when it ends.")
        self._loop.toggled.connect(self._scrubber.set_loop)
        self._scrubber.add_control(self._loop)
        self._block = PictureWithControls(
            self._view, self._scrubber, default_aspect=1.0
        )
        column.addWidget(self._block, 1)
        self._caption = QLabel("")
        self._caption.setObjectName("export_preview_caption")
        self._caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption.setWordWrap(True)
        column.addWidget(self._caption)
        self._plays_as = QLabel("")
        self._plays_as.setObjectName("export_preview_plays_as")
        self._plays_as.setWordWrap(True)
        column.addWidget(self._plays_as)
        hint = QLabel(_HINT)
        hint.setObjectName("export_preview_hint")
        hint.setWordWrap(True)
        hint.setEnabled(False)
        column.addWidget(hint)
        return box

    def _build_folder(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Folder:"))
        self._folder = QLineEdit(str(self._suggested_folder()))
        self._folder.setObjectName("export_folder")
        self._folder.setToolTip(
            "Each ticked visual gets a subfolder of this folder, named for the visual."
        )
        self._folder.textChanged.connect(lambda _t: self.refresh())
        row.addWidget(self._folder, 1)
        choose = QPushButton("Choose…")
        choose.setObjectName("export_choose_folder")
        choose.clicked.connect(self.choose_folder)
        row.addWidget(choose)
        return row

    # -- the folder ---------------------------------------------------------------- #
    def _settings_key(self) -> str:
        # A digest, not the path: a settings key takes neither a drive's colon nor a backslash.
        where = str(self._window.experiment_dir())
        return FOLDER_KEY + hashlib.sha1(where.encode("utf-8")).hexdigest()[:16]

    def _suggested_folder(self) -> Path:
        origin = self._window.origin()
        name = safe_name(
            self._window.title_text()
            if origin is None
            else f"{origin.run_id}_{origin.unit}_{origin.node}"
        )
        parent = QSettings().value(self._settings_key(), "", type=str)
        return (Path(parent) if parent else default_parent()) / name

    def choose_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Export the clips into…", self._folder.text()
        )
        if chosen:
            self._folder.setText(chosen)

    def folder(self) -> Path | None:
        text = self._folder.text().strip()
        return Path(text) if text else None

    def set_folder(self, path: Path | str) -> None:
        self._folder.setText(str(path))

    # -- what is asked ------------------------------------------------------------- #
    def ticked_visuals(self) -> list[str]:
        return [
            key
            for key, (tick, _show) in self._rows.items()
            if tick.isChecked() and tick.isEnabled()
        ]

    def write(self) -> str:
        return next(k for k, radio in self._writes.items() if radio.isChecked())

    def set_write(self, key: str) -> None:
        self._writes[key].setChecked(True)

    def selected_only(self) -> bool:
        return self._clips.currentIndex() == 1

    def set_selected_only(self, on: bool) -> None:
        if on and not self._window.selected_clips():
            on = False
        self._clips.setCurrentIndex(1 if on else 0)

    def clips(self) -> list:
        """The clips an export writes now, each its own span: the grid's in its order, or the
        picks in pick order."""
        if self.selected_only():
            return self._window.selected_clips()
        return self._window.shown_clips()

    def formats(self) -> list[export_api.ExportFormat]:
        out = []
        if self._mp4.isChecked():
            out.append(
                export_api.ExportFormat(
                    MP4,
                    self._mp4_width.currentData(),
                    float(self._mp4_fps.value()),
                    self._mp4_codec.codec(),
                )
            )
        if self._gif.isChecked():
            out.append(
                export_api.ExportFormat(
                    GIF, int(self._gif_width.value()), float(self._gif_fps.value())
                )
            )
        return out

    def trail_points(self) -> list[str] | None:
        return self._picker.value() if self._skeleton is not None else None

    def looks(self) -> tuple[export_api.ExportLook, ...]:
        return export_api.resolve_looks(
            self.ticked_visuals(),
            current=self._window.layers(),
            trail_points=self.trail_points(),
            point_count=len(self._skeleton[0]) if self._skeleton else None,
            align=self._align.isChecked(),
        )

    def plan(self) -> export_api.ExportPlan:
        """What Export would write now.

        Raises:
            ValueError: nothing to write (no visual, format or clip) — the reason Export is
                greyed.
        """
        clips = self.clips()
        return export_api.ExportPlan(
            looks=self.looks(),
            formats=tuple(self.formats()),
            write=self.write(),
            clips=tuple(self._wide(clips)),
            labels=tuple(montage_api.montage_labels(clips, blind=False)),
            stems=tuple(montage_api.clip_file_stem(c) for c in clips),
        )

    def current_label(self) -> str:
        return self._rows[export_api.CURRENT][0].text()

    def preview_key(self) -> str:
        return next(
            (k for k, (_t, show) in self._rows.items() if show.isChecked()),
            export_api.CURRENT,
        )

    # -- keeping it in step ---------------------------------------------------------- #
    def refresh(self) -> None:
        """Bring every label, greying and the summary in line with the grid and the choices
        (``MontageExport.sync`` calls this when its look, clips, context or picks change).
        """
        if not hasattr(self, "_summary"):
            return  # still building
        members = self._window.members()
        if members and members[0] != self._rank1 and not self._exporting:
            # The grid shows other clips (another cluster on INTERPRET): read theirs.
            self._rank1 = members[0]
            self._has_keypoints = None
            self._load()
        self._sync_current()
        self._sync_keypoints()
        self._sync_clips()
        self._sync_formats()
        self._sync_export()
        self._sync_preview()

    def looks_for_current(self) -> tuple[export_api.ExportLook, ...]:
        return export_api.resolve_looks(
            [export_api.CURRENT],
            current=self._window.layers(),
            trail_points=self.trail_points(),
            point_count=len(self._skeleton[0]) if self._skeleton else None,
            align=self._align.isChecked(),
        )

    def _sync_current(self) -> None:
        name = self.looks_for_current()[0].name
        self._rows[export_api.CURRENT][0].setText(f"Current preview settings  ({name})")

    def _sync_keypoints(self) -> None:
        known = self._has_keypoints
        for key, (tick, show) in self._rows.items():
            visual = export_api.VISUALS_BY_KEY.get(key)
            if visual is None or not visual.needs_keypoints:
                continue
            usable = bool(known)
            reason = CHECKING_KEYPOINTS if known is None else NO_KEYPOINTS
            for widget in (tick, show):
                widget.setEnabled(usable)
                widget.setToolTip("" if usable else reason)
            if not usable and show.isChecked():
                self._rows[export_api.CURRENT][1].setChecked(True)
        if self._skeleton is None:
            self._points.setEnabled(False)
            self._points.setText("every point")
            self._points.setToolTip(
                NO_KEYPOINTS
                if known is False
                else "Which keypoints leave a trail — "
                "once these clips' keypoints have been read."
            )
            return
        names = self._skeleton[0]
        chosen = self._picker.value()
        self._points.setEnabled(True)
        self._points.setText(
            f"all {len(names)} points"
            if chosen is None
            else f"{len(chosen)} of {len(names)} points"
        )
        self._points.setToolTip(
            "Which keypoints leave a trail, in every row with a keypoint trail."
        )

    def _sync_clips(self) -> None:
        shown = len(self._window.shown_clips())
        picked = len(self._window.selected_clips())
        self._clips.setItemText(0, f"All in the grid ({shown})")
        self._clips.setItemText(1, f"Picked only ({picked})")
        item = self._clips.model().item(1)
        item.setEnabled(picked > 0)
        item.setToolTip(
            "Only the clips picked in pick mode, in pick order."
            if picked
            else "No clip is picked: Export ▾ → Pick clips to export…, then click clips."
        )
        if picked == 0 and self._clips.currentIndex() == 1:
            self._clips.setCurrentIndex(0)

    def _sync_formats(self) -> None:
        for tick, widgets in (
            (self._mp4, (self._mp4_width, self._mp4_fps, self._mp4_codec)),
            (self._gif, (self._gif_width, self._gif_fps)),
        ):
            for widget in widgets:
                widget.setEnabled(tick.isChecked())
                widget.setToolTip(
                    self._tips[widget]
                    if tick.isChecked()
                    else f"Tick {tick.text()} to write it."
                )
        if self._native:
            # The typical clip's crop: one that crosses the arena is the whole frame wide, and
            # naming that size would misdescribe every other tile.
            typical = int(np.median(list(self._native.values())))
            self._mp4_width.setItemText(0, f"full size (~{typical} px tiles)")

    def _why_not(self) -> str | None:
        """Why Export cannot run now, or None when it can."""
        if self._exporting:
            return "Exporting…"
        if not self.ticked_visuals():
            return "Tick at least one visual to export."
        if not (self._mp4.isChecked() or self._gif.isChecked()):
            return "Tick MP4 or GIF (or both)."
        if not self.clips():
            return "There are no clips to export: the grid is empty."
        if self.folder() is None:
            return "Choose a folder to export into."
        return None

    def _sync_export(self) -> None:
        why = self._why_not()
        self._export.setEnabled(why is None)
        self._export.setToolTip(
            why or "Write every ticked visual, in every ticked format, into the folder."
        )
        if why is not None and not self._exporting:
            self._summary.setText(why)
            return
        if self._exporting:
            return
        try:
            plan = self.plan()
        except ValueError as exc:
            self._summary.setText(str(exc))
            return
        self._summary.setText(
            export_api.summary_text(plan) + "  ·  estimating size and time…"
        )
        self._estimate(plan)

    def summary(self) -> str:
        return self._summary.text()

    # -- reading and estimating, off the UI thread ------------------------------------ #
    def _load(self) -> None:
        """Which clips have keypoints, each clip's crop size, the footage's rate, and the
        skeleton the trail points are chosen from."""
        root, members = self._root, self._window.members()
        rank1 = self._wide_one(self._rank1)
        clips = self._wide([self._rank1, *(c for c in members if c != self._rank1)])
        need_skeleton = self._skeleton is None

        def work():
            keypoints = export_api.keypoints_available(  # off-thread (BackgroundCall)
                root.experiment_dir, members
            )
            native = export_api.native_tile_px(  # off-thread (BackgroundCall)
                root, clips
            )
            shown = root.subset(clips)
            footage_fps = shown.footage_fps
            skeleton = None
            if keypoints and need_skeleton:
                # A tile drawn with keypoints is how a montage learns its skeleton.
                single = root.subset([rank1])
                single.tile(  # off-thread (BackgroundCall)
                    0,
                    single.frame_at(0, 0),
                    montage_api.MontageLayers(mask=False, keypoints=True),
                    32,
                )
                found = root.skeleton()
                if found is not None:
                    skeleton = (tuple(found.names), tuple(found.pairs))
            return keypoints, native, footage_fps, skeleton

        call = BackgroundCall(work, receiver=self)
        call.done.connect(self._on_loaded)
        call.failed.connect(self._on_load_failed)
        call.start()

    def _on_loaded(self, result) -> None:
        keypoints, native, footage_fps, skeleton = result
        self._has_keypoints = bool(keypoints)
        self._native.update(native)
        self._footage_fps = float(footage_fps)
        if not self._mp4_fps_touched:
            self._mp4_fps.blockSignals(True)
            self._mp4_fps.setValue(round(self._footage_fps, 2))
            self._mp4_fps.blockSignals(False)
        if skeleton is not None and self._skeleton is None:
            self._skeleton = skeleton
            self._picker.set_skeleton(*skeleton)
        self.refresh()

    def _on_load_failed(self, exc: BaseException) -> None:
        self._has_keypoints = False
        self._status.setText(f"Could not read the clips: {exc}")
        self.refresh()

    def _on_mp4_fps(self, _value: float) -> None:
        self._mp4_fps_touched = True
        self.refresh()

    def _estimate(self, plan: export_api.ExportPlan) -> None:
        self._estimate_generation += 1
        generation = self._estimate_generation
        root, known, costs = self._root, dict(self._native), self._costs

        def work():
            missing = [c for c in plan.clips if c not in known]
            native = dict(known)
            if missing:
                found = export_api.native_tile_px(  # off-thread (BackgroundCall)
                    root, missing
                )
                native.update(found)
            estimate = export_api.estimate_export(  # off-thread (BackgroundCall)
                root, plan, native, costs
            )
            return native, estimate

        call = BackgroundCall(work, receiver=self)
        call.done.connect(
            lambda result, g=generation: self._on_estimated(g, plan, result)
        )
        call.failed.connect(lambda exc, g=generation: self._on_estimate_failed(g, exc))
        call.start()

    def _on_estimated(self, generation: int, plan, result) -> None:
        if generation != self._estimate_generation or self._exporting:
            return
        native, estimate = result
        self._native.update(native)
        self._summary.setText(export_api.summary_text(plan, estimate))
        self._sync_preview()  # a new context's crop sizes arrive with the estimate

    def _on_estimate_failed(self, generation: int, exc: BaseException) -> None:
        if generation == self._estimate_generation and not self._exporting:
            self._summary.setText(f"Could not estimate the export: {exc}")

    # -- the preview ------------------------------------------------------------------ #
    def _on_preview_pressed(self, key: str, on: bool) -> None:
        _tick, show = self._rows[key]
        show.setText("Previewing" if on else "Preview")
        if on:
            self._sync_preview()

    def _preview_format(self) -> tuple[export_api.ExportFormat, str]:
        """The format the preview plays as, and how to say which."""
        if self._mp4.isChecked() or not self._gif.isChecked():
            return (
                export_api.ExportFormat(
                    MP4,
                    self._mp4_width.currentData(),
                    float(self._mp4_fps.value()),
                    self._mp4_codec.codec(),
                ),
                "the MP4",
            )
        return (
            export_api.ExportFormat(
                GIF, int(self._gif_width.value()), float(self._gif_fps.value())
            ),
            "the GIF (MP4 is not ticked)",
        )

    def _preview_look(self) -> export_api.ExportLook:
        key = self.preview_key()
        return export_api.resolve_looks(
            [key],
            current=self._window.layers(),
            trail_points=self.trail_points(),
            point_count=len(self._skeleton[0]) if self._skeleton else None,
            align=self._align.isChecked(),
        )[0]

    def _sync_preview(self) -> None:
        rank1 = self._wide_one(self._rank1)
        if rank1 not in self._native:
            return  # the crop's size is still being read
        look = self._preview_look()
        fmt, which = self._preview_format()
        width = (
            self._native[rank1] + 2 * montage_api.GAP_PX
            if fmt.width is None
            else int(fmt.width)
        )
        tile = max(width - 2 * montage_api.GAP_PX, 16)
        fps = fmt.resolved_fps(self._footage_fps or montage_api.MONTAGE_MAX_FPS)
        label = montage_api.montage_labels([self._rank1], blind=False)[0]
        self._caption.setText(f"{label}      showing: {look.name}")
        self._plays_as.setText(f"Plays as {which} will: {width} px wide, {fps:g} fps.")
        spec = (look.layers, label, tile, fps)
        if self._preview_spec is not None and self._preview_spec[1:] == spec:
            return
        self._preview_generation += 1
        single = self._root.subset([rank1])
        duration = self._clip_duration(single)
        offsets = sample_offsets_ns(duration, fps)
        self._preview_spec = (self._preview_generation, *spec, single, offsets)
        self._scrubber.pause()
        self._scrubber.set_total(len(offsets))
        self._scrubber.set_playback_rate(fps)
        self._scrubber.set_timestamps(
            np.asarray(offsets, np.int64) + int(rank1.start_t_ns)
        )
        self._scrubber.set_frame(0)
        if len(offsets) > 1:
            self._scrubber.play()

    @staticmethod
    def _clip_duration(single) -> int:
        """The rank-1 clip's length — read by ``_load`` already, so this is a lookup."""
        return int(single.duration_ns)

    def _compute(self, step: int):
        spec = self._preview_spec
        if spec is None:
            return None
        generation, layers, label, tile, _fps, single, offsets = spec
        offset = offsets[min(max(int(step), 0), len(offsets) - 1)]
        view = frame_source.decode(  # off-thread (scrubber decoder)
            lambda: single.render(offset, layers, [label], columns=1, tile_px=tile)
        )
        return generation, view

    def _display(self, _step: int, result) -> None:
        if result is None:
            return
        generation, (picture, reason) = result
        if self._preview_spec is None or generation != self._preview_spec[0]:
            return
        if reason is not None:
            self._view.clear_frame(frame_source.unavailable_text(reason))
            return
        self._view.set_frame(_to_pixmap(picture))
        self._block.set_aspect(self._view.pixmap_size())

    def preview_shows(self):
        """``(clip, layers, tile_px, fps)`` the preview plays now — for a test. ``clip`` is
        as the file draws it, widened by the context."""
        spec = self._preview_spec
        if spec is None:
            return None
        return spec[5].clips[0], spec[1], spec[3], spec[4]

    def scrubber(self) -> FrameScrubber:
        return self._scrubber

    # -- exporting ---------------------------------------------------------------- #
    def export(self):
        """Write the plan into the folder, behind one bar with Cancel."""
        if self._why_not() is not None:
            return None
        plan = self.plan()
        folder = self.folder()
        QSettings().setValue(self._settings_key(), str(folder.parent))
        root, native, costs = self._root, dict(self._native), self._costs
        complete = all(c in native for c in plan.clips)
        session = self._window.pick_session() if self.selected_only() else None
        written: list[Path] = []
        total = len(plan.files())
        status_line = StageStatusLine()

        def work(progress):
            def report(stage, fraction, detail):
                # One status line, with its ETA, in the overlay's text.
                progress(status_line(stage, fraction, detail), fraction, "")

            paths = export_api.export_clips(  # off-thread (run_progress)
                root,
                plan,
                folder,
                native=native if complete else None,
                costs=costs,
                progress=report,
                on_written=written.append,
            )
            return paths, sum(p.stat().st_size for p in paths)

        def done(result) -> None:
            paths, size = result
            self._exporting = False
            note = f"Wrote {len(paths)} files ({size / 1e6:.1f} MB) into {folder}"
            codec = next((f.codec for f in plan.formats if f.kind == MP4), None)
            if codec is not None:
                note = with_codec_note(note, folder / "montage.mp4", codec)
            self._window.note_written(note, session=session)
            self._status.setText(note)
            self.accept()

        def stopped() -> None:
            self._exporting = False
            self._status.setText(
                f"Stopped: kept the {len(written)} finished file"
                f"{'' if len(written) == 1 else 's'} of {total} in {folder}; the one being "
                "written was removed."
            )
            self.refresh()

        def failed(exc: BaseException) -> None:
            self._exporting = False
            self._status.setText(
                f"The export stopped: {exc}. Kept {len(written)} finished files in {folder}."
            )
            self.refresh()

        self._exporting = True
        self._scrubber.pause()
        self._status.setText("")
        self._sync_export()
        return run_progress(
            self,
            work,
            done,
            label=f"Exporting {total} files…",
            on_cancel=stopped,
            on_error=failed,
            supersedes="montage export",
        )

    def is_exporting(self) -> bool:
        return self._exporting

    def _shut_down(self, *_args) -> None:
        self._preview_generation += 1
        self._preview_spec = None
        self._estimate_generation += 1
        self._scrubber.pause()
        self._root.close()
