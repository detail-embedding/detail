"""Stream dialogs: import, sync-to-video, and preview a stream (Section 3).

`AddStreamDialog` collects a name + importer + the importer's files and calls
`Recording.add_stream`. `SyncStreamDialog` scrubs the recording video to the frame
where a chosen stream anchor event occurs and sets the alignment offset from it
(`offset_ns = frame_t_ns(frame) - anchor_t_ns`). `StreamPreviewDialog` shows an
event-raster timeline linked to a scrubbable video frame (alignment applied), so
it doubles as a sync check. All are thin clients over the API (Tenet 2); Qt
lives only here (hard rule).
"""

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from detail.core.registries import STREAM_IMPORTERS
from detail.gui import sizing
from detail.gui import frame_source
from detail.gui.busy import decode_overlay, run_busy, run_progress
from detail.gui.dialog import Dialog
from detail.gui.image_view import ImageView, PictureWithControls
from detail.gui.importers.registry import (
    IMPORTER_GUIS,
    ConfigBand,
    import_importer_uis,
)
from detail.gui.lifetime import discard
from detail.gui.scrubber import FrameScrubber
from detail.gui.sizing import lines
from detail.gui.spin import FloatSpin
from detail.gui.status_color import ERROR, MUTED, set_label_color
from detail.gui.workers import BackgroundCall, StageProgressWorker
from detail.importers import import_stream_importers
from detail.storage import clocks

#: What "Save as…" does, in one sentence, including what it deliberately leaves out.
_SAVE_PRESET_HELP = (
    "Save this importer and its mapping under a name, so the experiment's other "
    "recordings can be imported the same way without picking columns again. The "
    "alignment offset is not saved: each recording has its own."
)


def _to_pixmap(image) -> QPixmap:
    """RGB ``(H, W, 3) uint8`` ndarray → QPixmap (mirrors gui/arena.py)."""
    height, width = image.shape[:2]
    return QPixmap.fromImage(
        QImage(image.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888)
    )


class _FilePicker(QWidget):
    """A read-only path field + Browse button; ``path`` is the chosen file or ''.

    ``chosen`` carries the new path, so a caller can react to the file itself (identifying
    it, reading its header) rather than waiting for submit.
    """

    chosen = pyqtSignal(str)

    def __init__(self, caption: str, file_filter: str, parent=None):
        super().__init__(parent)
        self._caption = caption
        self._filter = file_filter
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self._field = QLineEdit()
        self._field.setReadOnly(True)
        row.addWidget(self._field, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self._caption, "", self._filter)
        if path:
            self.set_path(path)

    def set_path(self, path: str) -> None:
        """Set the chosen file and announce it — what browsing does, minus the file dialog.

        One path in and one signal out, so a caller that supplies a file (a test, a drop, a
        preset) goes through exactly what a user's click goes through.
        """
        self._field.setText(path)
        self.chosen.emit(path)

    @property
    def path(self) -> str:
        return self._field.text().strip()


class AddStreamDialog(Dialog):
    """Import a time-series stream into a recording (Section 3).

    **The file comes first.** Every importer implements ``can_import``, so dropping a file in
    identifies it and the rest of the form follows; the picker stays, for overriding a wrong
    guess and for a file nothing claims. (That sniffing was implemented, tested, and called by
    nothing in ``detail/`` until this dialog used it.)

    Two bands under the picker. The **importer's** own config comes from its ``ImporterUI``
    band (the CSV column mapping; nothing at all for MED-PC or BORIS). The **dialog's** band is
    importer-agnostic and gated on declared properties, never a hand-written list of ids:
    *Labels are* on ``role == "behavior"``, *Alignment* on
    ``clocks.needs_alignment(clock)`` — so MED-PC gets offset-at-import for free, and a new
    importer inherits both the day it registers.

    Alignment is part of importing, not an errand afterwards. **Sync to video** opens the
    scrubber from here and returns an offset; nothing is written until Import. The Recording
    page keeps its own *Sync to video* / *Edit offset* for changing it later.

    On accept, calls ``Recording.add_stream``; the created stream's name is on
    ``self.stream_name``.
    """

    def __init__(self, recording, parent=None):
        super().__init__(parent)
        self._recording = recording
        self.stream_name: str | None = None
        self._offset_ns = 0
        self._auto_name = ""
        self._band: ConfigBand | None = None
        self._worker: BackgroundCall | None = None
        self.setWindowTitle("Add stream")
        self.resize(lines(34), lines(30))
        import_stream_importers()
        import_importer_uis()
        self._importers = dict(sorted(STREAM_IMPORTERS.items()))

        layout = self.body()

        top = QFormLayout()
        self._data = _FilePicker("Stream data file", "All files (*)")
        self._data.chosen.connect(self._on_file_chosen)
        top.addRow("Data file:", self._data)
        self._importer = QComboBox()
        for importer_id, cls in self._importers.items():
            self._importer.addItem(cls.display_name, importer_id)
        self._importer.currentIndexChanged.connect(self._on_importer_changed)
        top.addRow("Importer:", self._importer)

        # A mapping worked out on one recording, reused on the next. Only the *import* half
        # lives here; managing the saved set (delete / export / share) is CLI work, and
        # putting it in this dialog would make an import form a management surface too.
        preset_row = QHBoxLayout()
        self._preset = QComboBox()
        self._preset.currentIndexChanged.connect(self._on_preset_chosen)
        preset_row.addWidget(self._preset, 1)
        self._save_preset_btn = QPushButton("Save as…")
        self._save_preset_btn.setToolTip(_SAVE_PRESET_HELP)
        self._preset.setToolTip(
            "Apply a mapping saved from an earlier import: the time column, what it "
            "holds, and which columns to bring in. Only mappings saved under the "
            "importer selected above are listed."
        )
        self._save_preset_btn.clicked.connect(self._save_preset)
        preset_row.addWidget(self._save_preset_btn)
        # "Reuse a mapping", not "Saved setup": the row's job is what a user is trying to
        # do, and naming the noun left the whole row unreadable on first open -- a combo
        # with one meaningless entry beside a button with no stated object.
        top.addRow("Reuse a mapping:", preset_row)
        layout.addLayout(top)
        # The combo is filled by `_on_importer_changed` at the end of __init__, which owns
        # everything scoped to the importer. Filling it here too would read the file twice
        # and give the list a second owner.

        self._sniffed = QLabel()
        self._sniffed.setWordWrap(True)
        set_label_color(self._sniffed, MUTED)
        layout.addWidget(self._sniffed)

        self._aux_form = QFormLayout()
        self._aux_pickers: dict[str, _FilePicker] = {}
        layout.addLayout(self._aux_form)

        # The importer's own config band, swapped whenever the importer changes.
        self._band_holder = QVBoxLayout()
        layout.addLayout(self._band_holder, 1)

        bottom = QFormLayout()
        self._name = QLineEdit()
        self._name.textChanged.connect(lambda _text: self._refresh())
        bottom.addRow("Name:", self._name)
        # Behavior streams only: a re-ingested prediction must not be readable as ground
        # truth (§19). Greyed rather than hidden for a non-behavior importer, so the choice
        # is visible and its reason obvious.
        self._origin = QComboBox()
        self._origin.addItem("From the file (else ground truth)", None)
        self._origin.addItem("Ground truth — a human scored this", "ground_truth")
        self._origin.addItem("Predicted — a method produced this", "predicted")
        self._origin_label = QLabel("Labels are:")
        bottom.addRow(self._origin_label, self._origin)

        self._offset = FloatSpin("seconds")
        self._offset.setRange(-1e6, 1e6)
        self._offset.setToolTip(
            "video_t_ns = stream_t_ns + offset. Positive if the stream began after the "
            "video started."
        )
        self._offset.valueChanged.connect(self._on_offset_typed)
        self._sync_btn = QPushButton("Sync to video…")
        self._sync_btn.clicked.connect(self._sync_to_video)
        offset_row = QHBoxLayout()
        offset_row.addWidget(self._offset, 1)
        offset_row.addWidget(self._sync_btn)
        # The unit rides on the label, as everywhere else in the GUI -- no spin box in
        # this package sets a suffix, and a bare "3.48" says nothing about what it counts.
        self._offset_label = QLabel("Alignment (s):")
        bottom.addRow(self._offset_label, offset_row)
        layout.addLayout(bottom)

        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._box.button(QDialogButtonBox.StandardButton.Ok).setText("Import stream")
        self._box.accepted.connect(self._import)
        self._box.rejected.connect(self.reject)
        self.set_buttons(self._box)
        self._on_importer_changed()

    # -- the importer, and what it brings with it ---------------------------- #

    def _current_importer(self):
        importer_id = self._importer.currentData()
        return importer_id, self._importers[importer_id]

    def _on_file_chosen(self, path: str) -> None:
        """Identify the file, then hand it to whichever band is showing."""
        guess, claimed = self._sniff(path)
        if guess is not None and guess != self._importer.currentData():
            index = self._importer.findData(guess)
            if index >= 0:
                self._importer.setCurrentIndex(index)  # rebuilds the band
        self._describe_guess(path, guess, claimed)
        if self._band is not None:
            self._band.set_source(path)
        self._suggest_name()
        self._refresh()

    def _sniff(self, path: str) -> tuple[str | None, bool]:
        """``(importer_id, was_claimed)`` for ``path``.

        Several claimants is a real outcome, not an error: the user picks. Saying "several
        of these fit" beats picking one silently and importing under the wrong shape.

        **A file nothing claims falls back to the configurable importer**, which is the case
        this dialog exists for: ``can_import`` recognizes a CSV already shaped for DETAIL,
        and a CSV that needs a column mapping is by definition one it cannot recognize.
        Falling back to "the importer that can be told how to read a file" puts the mapping
        form in front of the user instead of an empty dialog. Derived from which importer
        declares a ``config_model`` rather than naming one, so it stays right if another
        configurable importer is added -- and gives up honestly if two are.
        """
        claimants = []
        for importer_id, cls in self._importers.items():
            try:
                if cls().can_import(Path(path)):
                    claimants.append(importer_id)
            except Exception:  # noqa: BLE001 - a sniff must never block the dialog
                continue
        if len(claimants) == 1:
            return claimants[0], True
        if claimants:
            return None, False  # several fit; the user picks
        configurable = [
            importer_id
            for importer_id, cls in self._importers.items()
            if getattr(cls, "config_model", None) is not None
        ]
        return (configurable[0], False) if len(configurable) == 1 else (None, False)

    def _describe_guess(self, path: str, guess: str | None, claimed: bool) -> None:
        """Say what the file was read as, and never imply more certainty than there is."""
        name = Path(path).name
        if claimed and guess is not None:
            self._sniffed.setText(
                f"Read as {self._importers[guess].display_name}. Change it above if that "
                "is not what this file is."
            )
        elif guess is not None:
            self._sniffed.setText(
                f"No importer recognized {name}, so it is set up as "
                f"{self._importers[guess].display_name} — say which column is time below, "
                "or change the importer above."
            )
        else:
            self._sniffed.setText(
                f"Could not tell what {name} is from its contents — check the importer "
                "above is the right one."
            )

    def _on_importer_changed(self) -> None:
        importer_id, cls = self._current_importer()
        self._rebuild_aux(cls)
        self._rebuild_band(importer_id)
        self._apply_capabilities(cls)
        # The saved-mapping list is scoped to the importer, so it is part of what an
        # importer change rebuilds -- not reloading it left the previous importer's
        # mappings on offer under the new one, which is the whole defect.
        self._reload_presets()
        self._suggest_name()
        self._refresh()

    def _rebuild_aux(self, cls) -> None:
        while self._aux_form.rowCount():
            self._aux_form.removeRow(0)
        self._aux_pickers = {}
        for aux in cls.aux_inputs:
            exts = " ".join(f"*{e}" for e in aux.extensions)
            picker = _FilePicker(aux.label, f"{aux.label} ({exts});;All files (*)")
            self._aux_form.addRow(f"{aux.label}:", picker)
            self._aux_pickers[aux.key] = picker

    def _rebuild_band(self, importer_id: str) -> None:
        """Swap in the picked importer's config band, or a spacer where it has none.

        The spacer is not decoration. This holder carries the dialog's stretch, so with an
        importer that contributes no band (MED-PC, BORIS) there is no fill child and the
        slack in a tall window falls on the two word-wrapped labels below — measured at
        233px against a natural 15px. Something here must always be able to absorb it.
        """
        while self._band_holder.count():
            item = self._band_holder.takeAt(0)
            discard(item.widget())
        self._band = None
        ui = IMPORTER_GUIS.get(importer_id)
        if ui is None:
            self._band_holder.addStretch(1)
            return
        self._band = ui.config_band(self)
        self._band.changed.connect(self._on_band_changed)
        self._band_holder.addWidget(self._band, 1)
        if self._data.path:
            self._band.set_source(self._data.path)

    def _apply_capabilities(self, cls) -> None:
        """Show or grey the dialog's own rows from what the importer *declares*.

        Read off `role` and `clock` rather than a list of importer ids, so this cannot go
        stale: an importer registered tomorrow gets the right rows with no edit here.
        """
        is_behavior = getattr(cls, "role", None) == "behavior"
        self._origin.setEnabled(is_behavior)
        self._origin_label.setEnabled(is_behavior)

        aligns = clocks.needs_alignment(getattr(cls, "clock", "stream_seconds"))
        has_video = bool(self._recording.video_parts)
        self._offset.setEnabled(aligns)
        self._offset_label.setEnabled(aligns)
        self._sync_btn.setEnabled(aligns and has_video)
        if not aligns:
            reason = "This stream is on the video clock — it is aligned at import."
        elif not has_video:
            reason = "Import a video part first"
        else:
            reason = ""
        for widget in (self._offset, self._sync_btn):
            widget.setToolTip(reason or widget.toolTip())
        if not aligns:
            self._offset.setValue(0.0)
            self._offset_ns = 0

    def _suggest_name(self) -> None:
        """Fill the name from the importer (or the band), keeping anything typed."""
        importer_id, cls = self._current_importer()
        from_band = self._band.suggested_name() if self._band is not None else ""
        suggestion = from_band or getattr(cls, "role", None) or importer_id
        if self._name.text().strip() in ("", self._auto_name):
            self._name.setText(suggestion)
        self._auto_name = suggestion

    # -- saved setups -------------------------------------------------------- #

    def _reload_presets(self, *, select: str | None = None) -> None:
        """The mappings saved under the importer now selected, and only those.

        **Scoped, not greyed.** The house rule is *grey out, don't hide*, and this is a
        deliberate exception with its reason (OQ-STR-5): a mapping saved under another
        importer is not a capability that does not apply here, it is an item from a
        different list. Listing them all meant picking one *switched the importer above* —
        a row that offers a mapping and delivers an importer change.
        """
        importer_id, cls = self._current_importer()
        self._preset.blockSignals(True)
        self._preset.clear()
        saved = [
            p
            for p in self._recording.list_stream_presets()
            if p.get("importer") == importer_id
        ]
        # An empty combo needs to read as *empty*, not broken: "— none —" beside a live
        # dropdown looks like a list that failed to load, and says nothing about how one
        # gets there. The Save-as button beside it is the answer, so the placeholder
        # points at it. It says *for this importer* because the list is scoped now: with
        # a MED-PC mapping saved and time-series selected, a bare "none saved yet" is a
        # lie about the experiment rather than a fact about the list.
        self._preset.addItem(
            "— none saved for this importer yet —" if not saved else "— none —", None
        )
        for preset in saved:
            # Prefixed with the importer even though the list is already scoped to it:
            # the scope is the thing that changed, and a bare name does not show it.
            self._preset.addItem(
                f"{cls.display_name}: {preset['name']}", preset["name"]
            )
        if select is not None:
            index = self._preset.findData(select)
            if index >= 0:
                self._preset.setCurrentIndex(index)
        self._preset.blockSignals(False)

    def _on_preset_chosen(self) -> None:
        name = self._preset.currentData()
        if name is None:
            return
        try:
            preset = self._recording.get_stream_preset(name)
        except ValueError as exc:
            QMessageBox.critical(self, "Saved setup", str(exc))
            return
        # No importer switch here any more. The list only offers mappings saved under the
        # importer already selected, so setting it was either a no-op or the defect: a
        # mapping from another importer silently changing what the form imports.
        if self._band is not None and preset.get("config"):
            self._band.set_config(preset["config"])
        if preset.get("offset_ns"):
            self._offset.setValue(preset["offset_ns"] / 1e9)
        # The preset's name beats the importer id as a stream name: it is the user's own
        # word for this kind of stream, and it is the same on every recording the preset is
        # applied to, which is what makes the streams comparable later. Both charsets are
        # `[A-Za-z0-9_-]`, so a valid preset name is always a valid stream name.
        if self._name.text().strip() in ("", self._auto_name):
            self._name.setText(name)
        self._auto_name = name
        self._refresh()

    def _save_preset(self) -> None:
        """Save the importer + mapping now on screen, under a name the user gives."""
        problem = self._band.problem() if self._band is not None else None
        if problem:
            QMessageBox.warning(self, "Save setup", problem)
            return
        # The charset is enforced in `storage/stream_presets.py` (`_ALLOWED`) and was only
        # discoverable by tripping over it -- a name with a space is refused after typing
        # it. The coupling is stated in the same breath because it is the reason the rule
        # cannot simply be relaxed: this name becomes the stream's name, and a stream name
        # is a directory name. See `_on_preset_chosen`.
        name, ok = QInputDialog.getText(
            self,
            "Save setup",
            "Name — letters, numbers, _ and - only (no spaces).\n"
            "It becomes the stream's name wherever this mapping is applied.",
        )
        name = name.strip()
        if not ok or not name:
            return
        existing = {p["name"] for p in self._recording.list_stream_presets()}
        if name in existing and (
            QMessageBox.question(
                self, "Save setup", f"A setup named {name!r} exists. Replace it?"
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        importer_id, _cls = self._current_importer()
        config = self._band.config() if self._band is not None else None
        try:
            self._recording.save_stream_preset(
                name, importer_id, config, overwrite=name in existing
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Save setup", str(exc))
            return
        self._reload_presets(select=name)

    # -- alignment ----------------------------------------------------------- #

    def _on_offset_typed(self, seconds: float) -> None:
        self._offset_ns = round(seconds * 1e9)

    def _sync_to_video(self) -> None:
        """Scrub the video to where this stream starts, before anything is written.

        Needs the parsed rows to place an anchor, so the file is read here — the same read
        the import will do, off the UI thread.
        """
        if not self._data.path:
            QMessageBox.warning(self, "Sync to video", "Choose a data file first.")
            return
        run_busy(
            self,
            lambda: self._parse_rows(),  # off-thread
            self._open_sync_dialog,
            label="Reading stream…",
            on_error=lambda exc: QMessageBox.critical(self, "Sync to video", str(exc)),
        )

    def _parse_rows(self):
        """``(columns, rows)`` for the current file + mapping, without writing anything."""
        importer_id, cls = self._current_importer()
        config = self._band.config() if self._band is not None else None
        result = cls().import_stream(
            Path(self._data.path),
            Path(self._data.path).parent,
            {k: Path(v.path) for k, v in self._aux_pickers.items() if v.path},
            config,
        )
        return result.columns, result.rows

    def _open_sync_dialog(self, parsed) -> None:
        columns, rows = parsed
        dialog = SyncStreamDialog.for_rows(self._recording, columns, rows, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._offset_ns = dialog.offset_ns
            self._offset.setValue(self._offset_ns / 1e9)

    # -- submit -------------------------------------------------------------- #

    def _on_band_changed(self) -> None:
        self._suggest_name()
        self._refresh()

    def _problem(self) -> str | None:
        """The one sentence to show, and to refuse the import with."""
        if not self._data.path:
            return "Choose a data file."
        for key, picker in self._aux_pickers.items():
            if not picker.path:
                return f"Choose the {key!r} file."
        if self._band is not None:
            band_problem = self._band.problem()
            if band_problem:
                return band_problem
        name = self._name.text().strip()
        if not name:
            return "The stream needs a name."
        if any(s.get("name") == name for s in self._recording.streams):
            return f"This recording already has a stream named {name!r}."
        return None

    def _refresh(self) -> None:
        problem = self._problem()
        self._status.setText(problem or "")
        set_label_color(self._status, ERROR if problem else MUTED)
        sizing.set_available(
            self._box.button(QDialogButtonBox.StandardButton.Ok),
            problem is None,
            problem or "",
        )
        # Nothing to save until a file makes the form describe a real import: offered on an
        # empty dialog it would store an importer name and no mapping, which reuses to
        # exactly what picking that importer by hand already does.
        band_problem = self._band.problem() if self._band is not None else None
        can_save = bool(self._data.path) and band_problem is None
        self._save_preset_btn.setEnabled(can_save)
        if not can_save:
            self._save_preset_btn.setToolTip(
                "Choose a data file and set up its mapping first"
            )
        else:
            self._save_preset_btn.setToolTip(_SAVE_PRESET_HELP)

    def _import(self) -> None:
        problem = self._problem()
        if problem is not None:
            QMessageBox.warning(self, "Add stream", problem)
            return
        name = self._name.text().strip()
        importer_id, _cls = self._current_importer()
        config = self._band.config() if self._band is not None else None
        try:
            self._recording.add_stream(
                name,
                raw_path=self._data.path,
                importer=importer_id,
                aux_paths={k: v.path for k, v in self._aux_pickers.items()},
                origin=self._origin.currentData() if self._origin.isEnabled() else None,
                config=config,
                offset_ns=self._offset_ns if self._offset.isEnabled() else 0,
            )
        except (FileNotFoundError, ValueError) as exc:
            QMessageBox.critical(self, "Add stream", str(exc))
            return
        self.stream_name = name
        self.accept()


class ImportBehaviorLabelsDialog(Dialog):
    """Batch-import behavior ground-truth labels onto existing recordings (Section 3).

    Pick a labels manifest (``schema_version: 1``; ``labels: {recording_id: file}``) + an
    importer; runs ``Experiment.import_behavior_labels`` off the UI thread and reports how
    many recordings imported vs were skipped. ``self.skipped`` holds the skipped list.
    """

    def __init__(self, experiment, parent=None):
        super().__init__(parent)
        self._experiment = experiment
        self._worker: StageProgressWorker | None = None
        self.skipped: list[tuple[str, str]] = []
        self.setWindowTitle("Import behavior labels")
        self.resize(560, 0)

        layout = self.body()
        form = QFormLayout()
        self._manifest = _FilePicker(
            "Labels manifest", "YAML (*.yaml *.yml);;All files (*)"
        )
        form.addRow("Manifest:", self._manifest)
        import_stream_importers()
        self._importer = QComboBox()
        for importer_id in ("behavior_csv", "boris"):
            self._importer.addItem(
                STREAM_IMPORTERS[importer_id].display_name, importer_id
            )
        form.addRow("Importer:", self._importer)
        self._name = QLineEdit("behavior")
        form.addRow("Stream name:", self._name)
        layout.addLayout(form)
        self._status = QLabel("")
        layout.addWidget(self._status)

        self._box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._box.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        self._box.accepted.connect(self._import)
        self._box.rejected.connect(self.reject)
        self.set_buttons(self._box)

    def _import(self) -> None:
        manifest = self._manifest.path
        if not manifest:
            QMessageBox.warning(
                self, "Import behavior labels", "Choose a manifest file."
            )
            return
        importer = self._importer.currentData()
        name = self._name.text().strip() or "behavior"
        sizing.set_available(
            self._box.button(QDialogButtonBox.StandardButton.Ok),
            False,
            "Importing…",
        )
        self._status.setText("Importing…")
        # The API always reported per recording; this dialog dropped it, so a batch across a
        # cohort was a status line reading "Importing…" until it ended (#916). Not cancellable:
        # a stop between recordings would leave the batch half-imported.
        self._worker = run_progress(
            self,
            lambda emit: self._experiment.import_behavior_labels(
                manifest,
                name=name,
                importer=importer,
                progress=lambda index, total, sid, fraction, _status: emit(
                    "importing",
                    (index + fraction) / max(total, 1),
                    f"{sid} ({index + 1} of {total})",
                ),
            ),
            self._on_done,
            label="Importing behavior labels…",
            on_error=self._on_failed,
        )

    def _on_done(self, skipped) -> None:
        self.skipped = skipped or []
        if self.skipped:
            detail = "\n".join(f"  {sid}: {err}" for sid, err in self.skipped[:20])
            QMessageBox.information(
                self,
                "Import behavior labels",
                f"Imported; {len(self.skipped)} recording(s) skipped:\n{detail}",
            )
        else:
            QMessageBox.information(
                self, "Import behavior labels", "Imported labels for every recording."
            )
        self.accept()

    def _on_failed(self, exc) -> None:
        self._box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self._status.setText("")
        QMessageBox.critical(self, "Import behavior labels", str(exc))


def _stream_columns(recording, name: str) -> list[str]:
    """A stream's canonical column names from its manifest, or ``[]`` if it has none."""
    entry = next((s for s in recording.streams if s.get("name") == name), None)
    return list((entry or {}).get("columns") or [])


class SyncStreamDialog(Dialog):
    """Set a stream's alignment offset by clicking the frame where it starts.

    Scrub to the video frame where the chosen anchor occurs; the offset is
    ``frame_t_ns(frame) - anchor_t_ns`` so ``video_t_ns = stream_t_ns + offset``. The
    default anchor is the stream's first row (stream t=0).

    **Two entry points, one dialog.** ``SyncStreamDialog(recording, name)`` edits a stream
    that exists and writes the offset itself; :meth:`for_rows` takes parsed rows instead, so
    the Add-stream dialog can sync *before* anything is written and read the answer off
    ``offset_ns``. Both compute the number the same way.
    """

    def __init__(
        self, recording, stream_name: str | None, parent=None, rows=None, columns=None
    ):
        super().__init__(parent)
        self._recording = recording
        self._stream = stream_name
        self.offset_ns = 0
        title = (
            f"Sync stream '{stream_name}' to video"
            if stream_name is not None
            else "Sync stream to video"
        )
        self.setWindowTitle(title)
        self.resize(720, 640)
        self._part_counts = recording.part_frame_counts
        self._total = sum(self._part_counts)
        self._frame = 0

        layout = self.body()
        # Zoomable: finding the frame a stream starts on means reading what is *in* the
        # frame — a lever, a nose crossing a line — at whatever size that detail happens to be.
        self._image = ImageView(zoomable=True)
        self._image.setMinimumHeight(360)

        # You are hunting for the frame where the stream starts; playing is how you find it.
        self._scrubber = FrameScrubber(playable=True)
        self._scrubber.set_total(self._total)
        if recording.video_fps:  # mtime-cached yaml read, safe on the UI thread
            self._scrubber.set_playback_rate(recording.video_fps)
        # The time beside the frame number, from the timestamp map — read off the UI thread,
        # since it parses every part's map. The row shows no time until it lands.
        self._times_call = BackgroundCall(
            recording.frame_timestamps, receiver=self  # off-thread
        )
        self._times_call.done.connect(self._on_times)
        self._times_call.failed.connect(
            lambda _exc: None
        )  # no time shown, never a guess
        self._times_call.start()
        self._scrubber.frame_changed.connect(self._on_scrub)
        self._scrubber.set_decoder(self._compute)  # frame decode off-thread
        self._scrubber.rendered.connect(self._display)  # on-thread setPixmap
        self._decode_busy = decode_overlay(self._scrubber, self._image)
        self._block = PictureWithControls(self._image, self._scrubber)
        layout.addWidget(self._block, 1)

        form = QFormLayout()
        self._anchor = QComboBox()
        self._anchor.currentIndexChanged.connect(self._update_offset)
        # Relabelled once the rows are in: a continuous stream's anchor is its start, not
        # an event, and calling it one invites a hunt for the event list it does not have.
        self._anchor_label = QLabel("Anchor event:")
        form.addRow(self._anchor_label, self._anchor)
        layout.addLayout(form)
        self._offset_label = QLabel()
        self._offset_label.setWordWrap(True)
        layout.addWidget(self._offset_label)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Set offset")
        box.accepted.connect(self._save)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

        # Anchors last: `_populate_anchors` writes `_offset_label`, so on the handed-in-rows
        # path (which is synchronous) it must run after the widgets exist. The store-backed
        # path defers through `run_busy` and so never noticed the order.
        if rows is not None:
            self._columns = list(columns or [])
            self._populate_anchors(rows)
        else:
            # Reading a large stream would freeze the dialog on open, so it loads off the UI
            # thread; the frame scrubber works meanwhile.
            self._columns = _stream_columns(recording, stream_name)
            run_busy(
                self,
                lambda: recording.stream_events(stream_name),  # off-thread
                self._populate_anchors,
                label="Loading stream…",
            )
        self._scrubber.set_frame(0)  # renders frame 0 + offset via _on_scrub

    def _on_times(self, t_ns) -> None:
        if len(t_ns) == self._total:
            self._scrubber.set_timestamps(t_ns)

    def _populate_anchors(self, events) -> None:
        """Fill the anchor picker once the stream's rows are available.

        A **continuous** stream has no events to anchor on: its rows are samples, and its
        columns after ``t_ns`` are channel values, not labels. Reading ``row[2]`` as a label
        listed sensor readings as anchor names on a multi-channel stream, one entry per
        sample, and raised ``IndexError`` on a single-channel one — where the only anchor
        that means anything is the stream's own start. So the label column is *found*, and
        a stream without one offers exactly one anchor.
        """
        self._anchor.blockSignals(True)  # one _update_offset at the end, not per row
        label_col = self._columns.index("label") if "label" in self._columns else None
        if label_col is None:
            first = int(events[0][0]) if events else 0
            self._anchor.addItem(f"stream start (t = {first / 1e9:.3f}s)", first)
            self._anchor_label.setText("Anchor:")
        else:
            for row in events:
                t_ns, label = row[0], row[label_col]
                self._anchor.addItem(f"{t_ns / 1e9:8.2f}s   {label}", t_ns)
        self._anchor.blockSignals(False)
        self._update_offset()

    def _locate(self, frame: int) -> tuple[int, int]:
        for part, count in enumerate(self._part_counts):
            if frame < count:
                return part, frame
            frame -= count
        return max(len(self._part_counts) - 1, 0), max(self._part_counts[-1] - 1, 0)

    def _compute(self, frame: int):
        part, local = self._locate(frame)
        return frame_source.decode(  # off-thread
            lambda: self._recording.get_frame(part, local)
        )

    def _display(self, _frame: int, payload) -> None:
        image, reason = payload
        if reason is not None:
            self._image.setText(frame_source.unavailable_text(reason))
            return
        # Full resolution: the view scales it, so zooming reaches real pixels.
        self._image.set_frame(_to_pixmap(image))
        self._block.set_aspect(self._image.pixmap_size())

    def _on_scrub(self, value: int) -> None:
        self._frame = value  # cheap; the frame image decodes async
        self._update_offset()

    def _computed_offset(self) -> int:
        anchor_t_ns = self._anchor.currentData() or 0
        return self._recording.frame_t_ns(self._frame) - int(anchor_t_ns)

    def _update_offset(self) -> None:
        offset = self._computed_offset()
        self._offset_label.setText(
            f"Offset = frame t_ns − anchor t_ns = {offset / 1e9:.3f}s   "
            "(video_t_ns = stream_t_ns + offset)"
        )

    def _save(self) -> None:
        self.offset_ns = self._computed_offset()
        if (
            self._stream is None
        ):  # opened before the stream exists; the caller applies it
            self.accept()
            return
        try:
            self._recording.set_stream_offset(self._stream, self.offset_ns)
        except ValueError as exc:
            QMessageBox.critical(self, "Sync stream", str(exc))
            return
        self.accept()

    @classmethod
    def for_rows(cls, recording, columns, rows, parent=None) -> "SyncStreamDialog":
        """Sync against rows that are parsed but not yet written (the import flow).

        Writes nothing: the caller reads ``offset_ns`` after an accepted ``exec()``.
        """
        return cls(recording, None, parent=parent, rows=rows, columns=columns)


class _EventStrip(QWidget):
    """An event-raster timeline: one lane per event type, a tick per event, a
    playhead. Click to jump (emits the clicked ``t_ns``). The ``video_range``
    band (if given) shades where the video covers the timeline."""

    clicked = pyqtSignal(int)  # t_ns at the click position
    _ROW_H = 16
    _LABEL_W = 150

    def __init__(self, lanes, events_by_lane, t_min, t_max, video_range=None):
        super().__init__()
        self._lanes = lanes
        self._events = events_by_lane
        self._t_min = int(t_min)
        self._t_max = max(int(t_max), int(t_min) + 1)
        self._video_range = video_range
        self._cursor_t = self._t_min
        self.setMinimumWidth(360)
        self.setFixedHeight(max(len(lanes), 1) * self._ROW_H + 4)

    def set_cursor_t(self, t_ns: int) -> None:
        self._cursor_t = int(t_ns)
        self.update()

    def _plot_w(self) -> int:
        return max(self.width() - self._LABEL_W, 1)

    def _x_for(self, t_ns: int) -> int:
        frac = (t_ns - self._t_min) / (self._t_max - self._t_min)
        return self._LABEL_W + int(frac * self._plot_w())

    def mousePressEvent(self, event) -> None:
        frac = (event.position().x() - self._LABEL_W) / self._plot_w()
        frac = min(max(frac, 0.0), 1.0)
        self.clicked.emit(int(self._t_min + frac * (self._t_max - self._t_min)))

    def paintEvent(self, _event) -> None:
        from detail.gui import plot_theme

        theme = plot_theme.current()
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(theme.surface))
        if self._video_range is not None:
            # The stretch the video covers, one step off the surface: a track, like the lane
            # extents in an exported ethogram.
            x0 = self._x_for(self._video_range[0])
            x1 = self._x_for(self._video_range[1])
            painter.fillRect(x0, 0, max(x1 - x0, 1), h, QColor(theme.grid))
        metrics = painter.fontMetrics()
        for r, lane in enumerate(self._lanes):
            y = r * self._ROW_H + 2
            color = theme.identity(r)
            painter.setPen(QColor(theme.text_secondary))
            painter.drawText(
                2,
                y + self._ROW_H - 4,
                metrics.elidedText(
                    lane, Qt.TextElideMode.ElideRight, self._LABEL_W - 6
                ),
            )
            painter.setPen(QColor(color))
            for t_ns in self._events.get(lane, ()):  # skip ticks outside the span
                if self._t_min <= t_ns <= self._t_max:
                    x = self._x_for(t_ns)
                    painter.drawLine(x, y, x, y + self._ROW_H - 3)
        painter.setPen(QColor(theme.text))
        cursor_x = self._x_for(self._cursor_t)
        painter.drawLine(cursor_x, 0, cursor_x, h)


class StreamPreviewDialog(Dialog):
    """Preview a stream: an event-raster timeline linked to a scrubbable video
    frame (alignment offset applied), with the events near the playhead listed.

    Works without video too (timeline only). The video band on the strip shows
    where the recording covers the timeline — events outside it have no frame.
    """

    _NOW_WINDOW_NS = 500_000_000  # ±0.5s around the playhead counts as "now"
    _TIME_STEPS = 1000  # slider resolution when there is no video

    def __init__(self, recording, stream_name: str, parent=None):
        super().__init__(parent)
        self._recording = recording
        self.setWindowTitle(f"Preview stream '{stream_name}'")
        self._layout = self.body()
        # Reading the stream + all frame timestamps scales with video length, so it
        # loads off the UI thread; _populate builds the view once the data is in.
        run_busy(
            self,
            lambda: self._load(recording, stream_name),
            self._populate,
            label="Loading stream…",
        )

    def _load(self, recording, stream_name: str) -> dict:
        """Off-thread: read the stream + frame timestamps + events; derive the plot data."""
        preview = recording.stream_preview(stream_name)  # off-thread
        part_counts = recording.part_frame_counts
        has_video = bool(part_counts)
        frame_t = (
            recording.frame_timestamps()  # off-thread
            if has_video
            else np.empty(0, np.int64)
        )
        # A continuous stream has no labels to make lanes from: its columns after `t_ns`
        # are channel values. Its samples still go on the strip, as one lane — which is
        # what a sync check needs (when does this stream have data, against the video)
        # and is the honest version of the raster that used to name a lane per sample
        # after that sample's value.
        if preview.has_labels:
            events = preview.events(aligned=has_video)
            lanes = preview.labels()
            header = preview.role or preview.shape
        else:
            channels = preview.channels
            lane = "samples"
            events = [(t, lane) for t in preview.times(aligned=has_video)]
            lanes = [lane]
            header = f"{preview.shape} · {', '.join(channels)}"
        events.sort()
        event_t = np.array([t for t, _ in events], dtype=np.int64)
        event_labels = [label for _, label in events]
        events_by_lane: dict[str, list[int]] = {lane: [] for lane in lanes}
        for t_ns, label in events:
            events_by_lane[label].append(t_ns)
        if has_video:
            video_end = int(frame_t[-1]) if len(frame_t) else 0
            video_range = (0, video_end)
            ev_lo = int(event_t.min()) if len(event_t) else 0
            ev_hi = int(event_t.max()) if len(event_t) else video_end
            t_min, t_max = min(0, ev_lo), max(video_end, ev_hi)
        else:
            video_range = None
            t_min = int(event_t.min()) if len(event_t) else 0
            t_max = int(event_t.max()) if len(event_t) else 1
        return {
            "part_counts": part_counts,
            "has_video": has_video,
            "frame_t": frame_t,
            "event_t": event_t,
            "event_labels": event_labels,
            "events_by_lane": events_by_lane,
            "lanes": lanes,
            "t_min": t_min,
            "t_max": t_max,
            "video_range": video_range,
            "offset_ns": preview.offset_ns,
            "header": header,
            "n_events": len(events),
        }

    def _populate(self, data: dict) -> None:
        self._part_counts = data["part_counts"]
        self._has_video = data["has_video"]
        self._frame_t = data["frame_t"]
        self._event_t = data["event_t"]
        self._event_labels = data["event_labels"]
        self._t_min, self._t_max = data["t_min"], data["t_max"]
        lanes = data["lanes"]
        layout = self._layout

        if self._has_video:
            self._image = ImageView(zoomable=True)
            self._image.setMinimumHeight(260)
        else:
            self._image = None

        offset_s = data["offset_ns"] / 1e9
        self._header = QLabel(
            f"{data['header']} · {data['n_events']} rows · offset {offset_s:+.3f}s"
            + ("" if self._has_video else "  (no video — stream-local time)")
        )
        self._now = QLabel()
        self._now.setWordWrap(True)

        self._strip = _EventStrip(
            lanes, data["events_by_lane"], self._t_min, self._t_max, data["video_range"]
        )
        self._strip.clicked.connect(self._on_strip_click)
        strip_h = max(len(lanes), 1) * _EventStrip._ROW_H + 6
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._strip)
        # A **maximum**, not a fixed height: an operant box with twenty lanes asks for more
        # than the window has, and a fixed height takes it from the video — which is the thing
        # being previewed. Tall enough to read several lanes, and it scrolls past that.
        scroll.setMaximumHeight(min(strip_h, 320))
        scroll.setMinimumHeight(min(strip_h, 3 * _EventStrip._ROW_H + 6))

        steps = sum(self._part_counts) - 1 if self._has_video else self._TIME_STEPS
        # Playable only with video. Without one there is no decoder at all — the scrubber is a
        # pure time cursor over the event raster, and a "step" is 1/1000 of the stream's span, a
        # duration that differs per stream. There is nothing for a rate to be 1x *of*.
        self._scrubber = FrameScrubber(
            label_formatter=self._pos_text,
            allow_frame_entry=self._has_video,
            playable=self._has_video,
        )
        self._scrubber.set_total(max(steps, 0) + 1)
        if self._has_video and len(self._frame_t) == steps + 1:
            # Already loaded for the cursor; without video the formatter prints the time instead.
            self._scrubber.set_timestamps(self._frame_t)
        self._scrubber.frame_changed.connect(self._on_scrub)
        if self._has_video:  # only the video case decodes a frame image
            self._scrubber.set_decoder(self._compute)
            self._scrubber.rendered.connect(self._display)
            self._decode_busy = decode_overlay(self._scrubber, self._image)
        # The video first, with the stretch: it is what this window is for, and reading a
        # frame against the events beside it is the whole task. The raster follows it, keeping
        # its lanes' own height until there are too many, and then scrolling.
        if self._image is None:
            layout.addWidget(self._scrubber)  # no video: the slider is a time cursor
        else:
            self._block = PictureWithControls(self._image, self._scrubber)
            layout.addWidget(self._block, 1)
        layout.addWidget(self._header)
        layout.addWidget(self._now)
        # Stretch on the raster too, capped above: a taller window shows more lanes rather
        # than only a bigger frame, and past its own full height the rest goes to the video.
        layout.addWidget(scroll, 1)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        box.accepted.connect(self.accept)
        self.set_buttons(box)
        self.resize(820, 620 if self._has_video else min(strip_h, 320) + 190)
        self._scrubber.set_frame(0)  # initial cursor + (video) frame image

    def _locate(self, frame: int) -> tuple[int, int]:
        for part, count in enumerate(self._part_counts):
            if frame < count:
                return part, frame
            frame -= count
        return max(len(self._part_counts) - 1, 0), max(self._part_counts[-1] - 1, 0)

    def _cursor_t_for_slider(self, value: int) -> int:
        if self._has_video:
            return int(self._frame_t[value])
        frac = value / self._TIME_STEPS
        return int(self._t_min + frac * (self._t_max - self._t_min))

    def _pos_text(self, value: int, last: int) -> str:
        if self._has_video:
            return f"{value} / {last}"
        return f"{self._cursor_t_for_slider(value) / 1e9:.1f}s"

    def _on_scrub(self, value: int) -> None:
        # cheap, immediate: move the raster cursor + "now" label; image decodes async
        cursor_t = self._cursor_t_for_slider(value)
        self._strip.set_cursor_t(cursor_t)
        self._update_now(cursor_t)

    def _compute(self, frame: int):
        part, local = self._locate(frame)
        return frame_source.decode(  # off-thread
            lambda: self._recording.get_frame(part, local)
        )

    def _display(self, _frame: int, payload) -> None:
        if self._image is None:
            return
        image, reason = payload
        if reason is not None:
            self._image.setText(frame_source.unavailable_text(reason))
            return
        # Full resolution: the view scales it, so zooming reaches real pixels.
        self._image.set_frame(_to_pixmap(image))
        self._block.set_aspect(self._image.pixmap_size())

    def _on_strip_click(self, t_ns: int) -> None:
        if self._has_video and len(self._frame_t):
            frame = int(np.searchsorted(self._frame_t, t_ns))
            self._scrubber.set_frame(min(max(frame, 0), len(self._frame_t) - 1))
        else:
            frac = (t_ns - self._t_min) / max(self._t_max - self._t_min, 1)
            self._scrubber.set_frame(int(frac * self._TIME_STEPS))

    def _update_now(self, cursor_t: int) -> None:
        lo = np.searchsorted(self._event_t, cursor_t - self._NOW_WINDOW_NS)
        hi = np.searchsorted(self._event_t, cursor_t + self._NOW_WINDOW_NS)
        near = [
            f"{self._event_t[i] / 1e9:.2f}s  {self._event_labels[i]}"
            for i in range(lo, hi)
        ]
        if near:
            self._now.setText("now (±0.5s):  " + "   ·   ".join(near[:6]))
        else:
            self._now.setText("now (±0.5s):  —")
