"""The one Export control, placed by the shell (#336).

**Why the shell places it.** Seven bespoke export dialogs, each on the page whose author needed
one, is how the capability ended up existing twice and reachable from neither page a user was on.
So the control is not a widget any page builds: it lives in the top bar beside `Info`, and its
subject comes from *navigation* — the same way `_set_page_help` works. A page cannot hand-list,
fall behind, or forget it, and a result page has export the day it becomes navigable.

**What a page does contribute is scope.** The catalogue is Qt-free and knows nothing about
widgets; a workspace knows which cluster is selected and which rows are picked. So a page that
has a selection implements `ExportScopeSurface`, and the shell asks — the same shape
`gui/editing.py` uses for unsaved edits, which is the existing "shell asks the page" precedent.

**A clip grid's exports are not here.** The clips a grid shows, as a video, GIFs or a CSV list,
are written from that grid's own **Export ▾** (``gui/montage_export.py``), which every clip grid
carries; the dialog once had a page-exports section for them, and nothing has offered one since.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Protocol, runtime_checkable

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.api import clip_render, export_targets, exporters
from detail.api.exporters import ExportScope, ExportSubject
from detail.api.experiment import Experiment
from detail.gui.dialog import Dialog
from detail.gui.recording_filter import with_filter
from detail.gui.video_codec import VideoCodecPicker
from detail.gui.workers import BackgroundCall
from detail.gui.spin import FloatSpin, IntSpin
from detail.gui.sizing import mono_font, set_available
from detail.gui.status_color import ERROR, set_label_color


@runtime_checkable
class ExportScopeSurface(Protocol):
    """A page that knows which part of its subject the user is looking at."""

    def export_scope(self) -> ExportScope | None:
        """The current selection, or ``None`` for "everything".

        Called when the Export menu opens, so it reads live state rather than whatever was
        selected when the page was built.
        """


def _ask(page, surface: type, method: str):
    """The first answer to ``method`` in ``page``'s tree, a surface's ``None`` covering its own
    subtree (see :func:`scope_of`)."""
    if page is None:
        return None
    if isinstance(page, surface):
        return getattr(page, method)()
    answered: list[QWidget] = []
    for child in page.findChildren(QWidget):
        if not isinstance(child, surface):
            continue
        if any(owner.isAncestorOf(child) for owner in answered):
            continue
        answer = getattr(child, method)()
        if answer is not None:
            return answer
        answered.append(child)
    return None


def scope_of(page) -> ExportScope | None:
    """The scope the shown page declares, searching the page and then its children.

    A workspace is usually embedded — `ClusterWorkspace` is a module's `result_widget` inside
    `RunDetailPage` — so asking only the top-level page would find nothing and silently export
    everything. Searching the tree means an embedder needs no forwarding method, which is what
    keeps a page from having to know it is exportable.
    """
    # A surface that answers `None` has said "everything" for its whole subtree. Searching on
    # into it reached the clip player inside a workspace's tab, so a workspace with nothing
    # selected exported whatever clip was on screen. `findChildren` is depth-first, so an
    # ancestor is always asked before anything under it.
    return _ask(page, ExportScopeSurface, "export_scope")


def open_export(parent, subject, scope=None, *, only=(), seed=None):
    """Open the one export dialog. Returns the submitted `ExportRun`, or ``None``.

    **The one place this dialog is constructed.** The top bar's `Export…` is one trigger and a
    clip player's `Export video…` is another, and the rule they both keep is that neither builds
    an export of its own — *"seven bespoke export dialogs, each on the page whose author needed
    one, is how the capability ended up existing twice and reachable from neither page a user
    was on."* A second *trigger* is not that failure; a second *implementation* is, which is
    what routing both through here prevents.
    """
    targets = exporters.targets_for(subject)
    if not targets:
        return None
    dialog = ExportDialog(subject, targets, scope, parent, only=only, seed=seed)
    if dialog.exec() and dialog.run_handle is not None:
        return dialog.run_handle
    return None


class ExportButton(QPushButton):
    """The top-bar `Export…`. Hidden when the navigated subject can produce nothing.

    Hidden rather than disabled, deliberately, and it is the one place this package departs
    from "grey out, don't hide": a disabled control is right when a capability exists but does
    not apply *here*, and every page in the app would then carry a permanently dead Export.

    One click, one dialog. It used to open a menu of targets, each leading to its own dialog —
    which meant the dialog could only ever narrow the *one* target you had already picked, so
    its "Everything" was a claim about a single file and read as a lie. Taking a screen away
    with you is one decision, not two.
    """

    #: The export run the dialog just submitted. The shell opens its page — the button has no
    #: business knowing what a run page is, and the same signal shape is what
    #: `_on_run_launched` / `_on_analysis_launched` already receive from the launch dialogs.
    launched = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__("Export…", parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Export this screen's data")
        self._subject: ExportSubject | None = None
        self._page = None
        self.hide()
        self.clicked.connect(self._open_dialog)

    def set_subject(self, subject: ExportSubject | None, page=None) -> None:
        """Point the control at what is on screen. ``None`` hides it."""
        self._subject = subject
        self._page = page
        # Shown when something can actually be written. A screen where every target is off
        # would otherwise carry a button whose only content is a list of things you cannot do.
        runnable = exporters.available_targets(subject) if subject is not None else []
        self.setVisible(bool(runnable))

    def _open_dialog(self) -> None:
        if self._subject is None:
            return
        handle = open_export(self, self._subject, scope_of(self._page))
        if handle is not None:
            # The result used to be discarded, which was right while the dialog *did* the
            # export. Now it submits one, and a run nobody is shown is a run nobody knows about.
            self.launched.emit(handle)


class ExportNotice(QFrame):
    """The strip that says an export started, so nothing has to hold you while it runs.

    **What this replaces.** The dialog used to close and the shell navigated to the new run's
    page — the move every other launch in this app makes, and the wrong one here. A run is the
    thing you came to start; an export is an errand off a screen you are *using*, and being
    taken away from that screen to watch a bar is a worse answer than being told where the bar
    is. A modal "export started" box is no better: it is still a window between you and the
    work, and it still has to be clicked away.

    So the answer sits where the click was — two inches under the **Export…** button — says what
    it started and where it is going, and offers the one route worth offering. It clears on the
    next navigation (`MainWindow._show`), because by then you have gone to look at something and
    the notice has done its job.

    **It does not draw a bar.** Live progress belongs to the run's own page, and a second
    ``event -> render`` mapping is the thing DESIGN_PLAN.md's progress rules forbid — it would
    also be the one nobody tests. The text is therefore written to stay true for as long as it
    is on screen: an export that started *did* start, whatever it is doing now.

    **Imports use it too** (`say`). An import finishing is the same kind of news — an errand whose
    result is listed under Active processes — and it used to arrive as a status-bar message at
    the bottom edge of the window, where the user who reported it did not see it. One strip for
    both, rather than a second notice written next to whichever surface was being worked on.
    """

    #: Clicked "Show Active processes". The shell switches mode — the notice has no business
    #: knowing what a mode is, the same split `ExportButton.launched` already uses.
    show_processes = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        self._label = QLabel("")
        self._label.setWordWrap(True)
        # Set here rather than inherited: `_show` makes a *page's* text selectable, and this is
        # shell chrome, which never passes through it. The destination is the whole reason to
        # read this line and it is a path — a path you cannot copy is half a path.
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._label, 1)
        button = QPushButton("Show Active processes →")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(self.show_processes)
        layout.addWidget(button)
        self.hide()

    def announce(self, what: str, where: str) -> None:
        """Say what export was started and where it is writing, then show the strip."""
        wrote = f" → {where}" if where else ""
        self.say(
            f"Export started — {what}{wrote}. It keeps going if you close this window; "
            f"watch it under Active processes."
        )

    def say(self, message: str) -> None:
        """Replace the strip's text with ``message`` and show it."""
        self._label.setText(message)
        self.show()


class ExportDialog(Dialog):
    """Pick what to take away, where to put it, and how much of it.

    One dialog listing everything this screen can produce, rather than one dialog per target
    reached through a menu. That was the shape's real defect: having already chosen a single
    target, the dialog's "everything" could only mean *all of that one file*, which reads as a
    lie next to a checklist that plainly is not everything.

    **The dialog submits a run; it does not export.** It used to hold a `run_progress` overlay
    for the whole export, which is the thing #354 exists to stop — a forty-minute clip export is
    not something to keep a modal window open for. Ticking the boxes is the decision; carrying it
    out is a run with its own page, which survives this window closing and the app closing after
    it. So the last thing this dialog does is hand a handle up and get out of the way.
    """

    def __init__(
        self,
        subject,
        targets,
        scope,
        parent=None,
        *,
        only=(),
        seed=None,
    ) -> None:
        """``only`` ticks just those target ids; ``seed`` supplies options no control offers.

        Both exist for a trigger that already knows what it wants — the clip player's
        **Export video…** knows the target, the span, the subject and the flavour on screen, so
        the dialog opens on that answer rather than on a checklist the user has to redo. It is
        still *this* dialog: the destination, the knobs and the submit are the ones every other
        export goes through, which is the part of "the shell places the control" that matters.
        """
        super().__init__(parent)
        self._subject = subject
        self._targets = list(targets)
        self._scope = scope or ExportScope()
        #: The page's scope minus its recordings, which the Recordings list below owns: two
        #: controls for one choice is the class `RunResults` was written to end.
        self._scope_rest = dataclasses.replace(self._scope, recording_ids=())
        self._only = tuple(only)
        self._seed = dict(seed or {})
        self._busy = False
        self._space_ok = True
        self._estimate = None
        self._estimate_generation = 0
        #: Set when the export has been submitted — the shell reads it and opens the run.
        #: Replaces the old `written`, which was the finished destination: at the moment
        #: this dialog closes there is no destination yet, only a run heading for one.
        self.run_handle = None
        self.setWindowTitle("Export")

        layout = self.body()
        form = QFormLayout()

        self._out = QLineEdit()
        browse = QPushButton("Choose")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self._out, 1)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Save to:", holder)

        # The folder this will create, spelled out before anything is written. An export names
        # itself rather than scattering files into a directory you chose, so two exports never
        # collide and one is still identifiable months later on somebody else's machine.
        self._folder = QLabel(exporters.folder_name(subject) + "/")
        self._folder.setFont(_mono())
        form.addRow("Creates:", self._folder)

        # Every target this kind of screen can produce, whether or not it can run right now.
        # **Grey out, don't hide** (gui_design.md §3.1): a hidden row leaves the reader unable
        # to tell "DETAIL cannot export my streams" from "this recording has no streams yet",
        # and only the second is something they can act on.
        self._include: dict[str, QCheckBox] = {}
        box = QVBoxLayout()
        for target in self._targets:
            reason = exporters.unavailable_reason(target, subject)
            check = QCheckBox(
                target.label if reason is None else f"{target.label} — {reason}"
            )
            if reason is None:
                # The bundle is opt-in; data is the default. A caller that named `only` has
                # already made this choice, so honour it rather than re-asking.
                check.setChecked(
                    target.id in self._only if self._only else not target.portable
                )
                check.setToolTip(
                    f"{target.artifact}\nAlso writes: "
                    + ", ".join(target.included_files())
                )
            else:
                check.setEnabled(False)
                check.setToolTip(target.artifact)
            self._include[target.id] = check
            box.addWidget(check)
        holder = QWidget()
        holder.setLayout(box)
        form.addRow("Include:", holder)

        # Only when there is something to choose between. A picker with one entry is a control
        # that cannot be operated, and its single entry reads as a claim about the whole form.
        self._scope_pick = None
        if not self._scope_rest.is_all() and any(t.scopes for t in self._targets):
            self._scope_pick = QComboBox()
            self._scope_pick.addItem(self._scope_rest.describe().capitalize(), True)
            self._scope_pick.addItem("Everything in this run", False)
            form.addRow("Of:", self._scope_pick)

        # Which recordings. A processing run spans many, and what a video or a CSV costs is
        # *per recording*: an export that wrote one video per recording of a large run
        # was over a hundred GB when one was wanted. All are ticked unless the page on screen says it is showing one (as
        # it does for a cluster or a selection), and the estimate below follows the ticks.
        # Greyed, not hidden, while no ticked target is per-recording.
        self._recording_list = None
        self._recording_ids: list[str] = []
        if subject.kind == "run" and any(
            "recordings" in t.scopes and self._include[t.id].isEnabled()
            for t in self._targets
        ):
            self._recording_ids = exporters.recordings_of(subject)
        if len(self._recording_ids) > 1:
            experiment = Experiment(subject.experiment_dir)
            names = experiment.recording_names()
            wanted = set(self._scope.recording_ids) or set(self._recording_ids)
            self._recording_list = QListWidget()
            self._recording_list.setMaximumHeight(140)
            for recording_id in self._recording_ids:
                item = QListWidgetItem(names.get(recording_id, recording_id))
                item.setData(Qt.ItemDataRole.UserRole, recording_id)
                item.setToolTip(recording_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if recording_id in wanted
                    else Qt.CheckState.Unchecked
                )
                self._recording_list.addItem(item)
            self._recordings_count = QLabel()
            select_all = QPushButton("All")
            select_none = QPushButton("None")
            select_all.clicked.connect(lambda: self._tick_recordings(True))
            select_none.clicked.connect(lambda: self._tick_recordings(False))
            head = QHBoxLayout()
            head.addWidget(self._recordings_count, 1)
            head.addWidget(select_all)
            head.addWidget(select_none)
            column = QVBoxLayout()
            column.setContentsMargins(0, 0, 0, 0)
            column.addLayout(head)
            column.addWidget(with_filter(self._recording_list, experiment))
            holder = QWidget()
            holder.setLayout(column)
            form.addRow("Recordings:", holder)
            self._recordings_holder = holder
            self._recording_list.itemChanged.connect(self._changed)
            self._sync_recordings_row()

        # The bundle's own knobs. They existed on the API from the start and no surface
        # exposed them, so the GUI always wrote a slim bundle and nothing on screen said so —
        # a default that only the person reading the source could know about.
        self._bundle_video = self._bundle_datasets = self._bundle_gzip = None
        if any(t.portable and self._include[t.id].isEnabled() for t in self._targets):
            self._bundle_video = QCheckBox("Source video")
            self._bundle_video.setToolTip(
                "Video DETAIL imported a copy of. A recording whose video is referenced in "
                "place cannot travel in a bundle — those files live outside the experiment — "
                "and the bundle records which recordings they were, so whoever opens it knows "
                "which video root to register."
            )
            self._bundle_datasets = QCheckBox("Training datasets")
            self._bundle_datasets.setToolTip(
                "Exported image datasets. Large, and re-generatable from the runs that are in "
                "the bundle either way."
            )
            # Carried over from the Experiment page's own export dialog when that one was
            # retired for this one (#916), with the measured reason not to tick it in the tip.
            self._bundle_gzip = QCheckBox("Compress (gzip)")
            self._bundle_gzip.setToolTip(
                "Much slower than an uncompressed bundle and barely smaller, "
                "because DETAIL's data files are compressed "
                "already. Worth it only for a slow link and a bundle of mostly text."
            )
            box = QVBoxLayout()
            box.addWidget(self._bundle_video)
            box.addWidget(self._bundle_datasets)
            box.addWidget(self._bundle_gzip)
            holder = QWidget()
            holder.setLayout(box)
            form.addRow("Bundle also packs:", holder)

        self._renders: dict[str, QCheckBox] = {}
        self._crop = self._size = self._max_per_cluster = None
        clip_targets = [
            t for t in self._targets if t.renders and self._include[t.id].isEnabled()
        ]
        if clip_targets:
            box = QVBoxLayout()
            for render in clip_targets[0].renders:
                check = QCheckBox(_RENDER_LABELS[render])
                check.setChecked(render == clip_render.OVERLAY)
                self._renders[render] = check
                box.addWidget(check)
            holder = QWidget()
            holder.setLayout(box)
            form.addRow("Video for each:", holder)

            self._crop = QCheckBox("Crop to the subject instead of the whole frame")
            self._crop.setToolTip(
                "The full field of view is the default: a steady camera, where the "
                "surroundings are half the answer. A crop keeps the pose at a constant "
                "scale, which is what makes bouts comparable side by side."
            )
            form.addRow("", self._crop)

            self._size = IntSpin()
            self._size.setRange(64, 4096)
            self._size.setValue(640)
            self._size.setToolTip(
                "Longest side of each clip; the source aspect is kept. Barely affects run "
                "time — decoding the full-resolution source dominates."
            )
            form.addRow("Clip size (px):", self._size)

            self._max_per_cluster = IntSpin()
            self._max_per_cluster.setRange(0, 100000)
            self._max_per_cluster.setSpecialValueText("(every bout)")
            self._max_per_cluster.setToolTip(
                "Cap the clips per cluster — a quick look at a large result."
            )
            form.addRow("Clips per cluster:", self._max_per_cluster)

        # §26 keypoint exports: the source and the cut-off are decided once here and reach
        # every keypoint format in the run, so a CSV and an NWB of the same export agree.
        self._kp_source = self._kp_cutoff = self._kp_blank = self._kp_overlay = None
        self._build_keypoint_knobs(form)

        # Only where a chosen target writes video, for the same reason as the figures below.
        self._codec = None
        if any(
            t.writes_video and self._include[t.id].isEnabled() for t in self._targets
        ):
            self._codec = VideoCodecPicker()
            form.addRow("Video codec:", self._codec)

        # Only where a chosen target draws a figure: a control for nothing is a question the
        # user cannot answer.
        self._dark_figures = None
        if any(
            t.figure_files and self._include[t.id].isEnabled() for t in self._targets
        ):
            self._dark_figures = QCheckBox("Dark background (for slides)")
            self._dark_figures.setToolTip(
                "Figures are drawn on white for print unless this is ticked. Dark is the same "
                "design on a dark surface — for a talk, not a paper."
            )
            form.addRow("Figures:", self._dark_figures)

        # What it will cost, before it is spent. Asked of the same catalogue `detail export run`
        # asks, with the same options, off the UI thread (a video target times a few dozen
        # frames), and it follows every control above.
        self._estimate_label = QLabel("")
        self._estimate_label.setWordWrap(True)
        self._estimate_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("Estimate:", self._estimate_label)
        self._estimate_timer = QTimer(self)
        self._estimate_timer.setSingleShot(True)
        self._estimate_timer.setInterval(300)
        self._estimate_timer.timeout.connect(self._start_estimate)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        buttons.accepted.connect(self._export)
        buttons.rejected.connect(self.reject)
        self._buttons = buttons
        self.set_buttons(buttons)

        # Connected last: every control exists, and none fires while the form is being built.
        for check in (*self._include.values(), *self._renders.values()):
            check.toggled.connect(self._changed)
        # The bundle's three boxes change what it packs and how long that takes, so the estimate
        # follows them. They were left out of this list, and ticking "Source video" left the
        # estimate saying what a slim bundle costs.
        for widget in (
            self._crop,
            self._dark_figures,
            self._kp_blank,
            self._kp_overlay,
            self._bundle_video,
            self._bundle_datasets,
            self._bundle_gzip,
        ):
            if widget is not None:
                widget.toggled.connect(self._changed)
        for spin in (
            self._size,
            self._max_per_cluster,
            self._kp_cutoff,
        ):
            if spin is not None:
                spin.valueChanged.connect(self._changed)
        if self._kp_source is not None:
            self._kp_source.currentIndexChanged.connect(self._changed)
        if self._codec is not None:
            self._codec.changed.connect(self._changed)
        if self._scope_pick is not None:
            self._scope_pick.currentIndexChanged.connect(self._changed)
        self._out.textChanged.connect(self._changed)
        self._changed()

    def _build_keypoint_knobs(self, form: QFormLayout) -> None:
        """The keypoint targets' shared knobs, only when one of them can run here."""
        from detail.api import export_targets

        keypoint_targets = [
            t
            for t in self._targets
            if t.id in export_targets.KEYPOINT_TARGETS
            and self._include[t.id].isEnabled()
        ]
        if not keypoint_targets:
            return
        record = export_targets.run_record(self._subject) or {}
        if record.get("module_id") == "keypoint_track":
            self._kp_source = QComboBox()
            self._kp_source.addItem("Tracking", "track")
            self._kp_source.addItem(
                "Detection, before tracking and smoothing", "detect"
            )
            self._kp_source.setToolTip(
                "Every keypoint file in this export reads one source. Tracking is this run's "
                "final points. Detection is the points before tracking settled them; export "
                "both to see what tracking changed."
            )
            form.addRow("Points from:", self._kp_source)
        self._kp_cutoff = FloatSpin()
        self._kp_cutoff.setRange(-1.0, 1.0)
        self._kp_cutoff.setSingleStep(0.05)
        self._kp_cutoff.setSpecialValueText("(the method's default)")
        self._kp_cutoff.setValue(self._kp_cutoff.minimum())
        self._kp_cutoff.setToolTip(
            "A point scoring under this is written as not visible, and the video does not "
            "draw it. Its coordinates are kept unless the box below is ticked. Leave at the "
            "method's default to use the detect method's own cut-off."
        )
        form.addRow("Score threshold:", self._kp_cutoff)
        self._kp_blank = QCheckBox("Omit points below score threshold")
        self._kp_blank.setToolTip(
            "Off, a point under the cut-off keeps its coordinates and only the visible "
            "column says it is under, so another threshold is a filter rather than a re-run. "
            "On, it is NaN in every format, which is what a pose tool that cannot read a "
            "visibility column wants."
        )
        form.addRow("", self._kp_blank)
        blank_note = QLabel("Off: kept, marked not visible. On: written as empty.")
        blank_note.setWordWrap(True)
        form.addRow("", blank_note)
        if any(t.id == "keypoints_video" for t in keypoint_targets):
            self._kp_overlay = QCheckBox("Video: show masks and boxes")
            self._kp_overlay.setToolTip(
                "The masks and boxes under the points, as the run's viewer shows them: a "
                "track run's tracked animals, or every detection of a detect run's "
                "detection run. Off, the points alone go on the footage."
            )
            form.addRow("", self._kp_overlay)
            overlay_note = QLabel(
                "The tracking run's masks and boxes, drawn under the points."
            )
            overlay_note.setWordWrap(True)
            form.addRow("", overlay_note)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Save the export into…")
        if path:
            self._out.setText(path)

    def _chosen(self) -> list:
        return [t for t in self._targets if self._include[t.id].isChecked()]

    # -- which recordings ------------------------------------------------------------- #
    def _ticked_recordings(self) -> list[str]:
        if self._recording_list is None:
            return []
        items = (
            self._recording_list.item(i) for i in range(self._recording_list.count())
        )
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in items
            if item.checkState() == Qt.CheckState.Checked
        ]

    def _tick_recordings(self, ticked: bool) -> None:
        """Tick or clear every recording at once, as one change rather than one per row."""
        state = Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked
        blocked = self._recording_list.blockSignals(True)
        for i in range(self._recording_list.count()):
            self._recording_list.item(i).setCheckState(state)
        self._recording_list.blockSignals(blocked)
        self._changed()

    def _per_recording_chosen(self) -> bool:
        return any("recordings" in t.scopes for t in self._chosen())

    def _sync_recordings_row(self) -> None:
        """Say how many are ticked, and grey the list while nothing ticked above uses it."""
        if self._recording_list is None:
            return
        applies = self._per_recording_chosen()
        self._recordings_holder.setEnabled(applies)
        ticked, total = len(self._ticked_recordings()), len(self._recording_ids)
        text = f"all {total}" if ticked == total else f"{ticked} of {total} selected"
        if not applies:
            text += " (nothing ticked above is written per recording)"
        self._recordings_count.setText(text)

    def _final_scope(self) -> ExportScope:
        """The scope the export will run with: the page's, the picker's, and the ticked recordings.

        Recordings are only a scope when it is a real choice (some unticked) and a ticked target
        is per recording, so "all of them" stays the empty scope that means everything and a
        manifest never says "3 recordings" about an export of a run-wide report.
        """
        scope = self._scope_rest
        if self._scope_pick is not None and not self._scope_pick.currentData():
            scope = ExportScope()
        if not self._per_recording_chosen():
            return scope
        if self._recording_list is not None:
            ticked = self._ticked_recordings()
            if len(ticked) != len(self._recording_ids):
                return dataclasses.replace(scope, recording_ids=tuple(ticked))
        elif self._scope.recording_ids:
            # No list to show (one recording, or an analysis run): keep what the page named.
            return dataclasses.replace(scope, recording_ids=self._scope.recording_ids)
        return scope

    # -- what it will cost ------------------------------------------------------------- #
    def _changed(self, *_args) -> None:
        """Something that changes the answer changed: ask again once the user has paused."""
        self._sync_recordings_row()
        self._estimate_label.setText("Estimating…")
        self._estimate_timer.start()

    def _start_estimate(self) -> None:
        chosen = self._chosen()
        if not chosen:
            self._estimate_generation += 1
            self._on_no_estimate("Tick something to export.")
            return
        subject = self._subject
        target_ids = [t.id for t in chosen]
        scope = self._final_scope()
        options = self._options()
        parent = self._out.text().strip() or None
        self._estimate_generation += 1
        generation = self._estimate_generation
        # Off the UI thread: a video target times a few dozen frames, and the drive probe writes
        # 16 MiB. `receiver=self` drops a result that lands after the dialog is gone.
        call = BackgroundCall(
            lambda: Experiment(subject.experiment_dir).estimate_export(
                subject.kind, subject.id, target_ids, parent, scope=scope, **options
            ),
            receiver=self,
        )
        self._estimate_call = call  # kept alive until it has emitted
        call.done.connect(lambda result, g=generation: self._on_estimated(g, result))
        call.failed.connect(lambda exc, g=generation: self._on_estimate_failed(g, exc))
        call.start()

    def _on_estimated(self, generation: int, result) -> None:
        if generation != self._estimate_generation:
            return  # superseded by a later change
        self._estimate = result
        text = exporters.describe_estimate(result)
        if result.free_bytes is not None:
            text += f" · {exporters.format_size(result.free_bytes)} free there"
        fits = result.fits
        if fits is False:
            text += ": it will not fit. Choose another folder, or untick some recordings or targets."
        self._estimate_label.setText(text)
        self._estimate_label.setToolTip(
            "\n".join(
                f"{self._label_of(target_id)}: {exporters.describe_one(one)}"
                + (f"\n    {one.basis}" if one.basis else "")
                for target_id, one in result.per_target
            )
            + "\n\nEstimates, not promises: sizes are counted or timed on a sample, and "
            "video tends to read high."
        )
        self._tint_estimate(ERROR if fits is False else None)
        self._space_ok = fits is not False
        self._sync_export_enabled()

    def _on_estimate_failed(self, generation: int, exc: BaseException) -> None:
        if generation != self._estimate_generation:
            return
        self._on_no_estimate(f"could not be estimated ({exc})")

    def _on_no_estimate(self, text: str) -> None:
        """No numbers: the export is still allowed, since a missing estimate is not a refusal."""
        self._estimate = None
        self._estimate_label.setText(text)
        self._estimate_label.setToolTip("")
        self._tint_estimate(None)
        self._space_ok = True
        self._sync_export_enabled()

    def _tint_estimate(self, color) -> None:
        """The estimate in a status colour, or back to the theme's own text colour."""
        if color is None:
            self._estimate_label.setPalette(QPalette())
        else:
            set_label_color(self._estimate_label, color)

    def _label_of(self, target_id: str) -> str:
        return next((t.label for t in self._targets if t.id == target_id), target_id)

    def _sync_export_enabled(self) -> None:
        """Export is offered unless it is being submitted or the estimate says it cannot fit."""
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            self._space_ok and not self._busy
        )

    def _options(self) -> dict:
        options: dict = {}
        if self._renders:
            options["renders"] = tuple(
                r for r, check in self._renders.items() if check.isChecked()
            )
            options["crop"] = self._crop.isChecked()
            options["size"] = self._size.value()
            options["max_per_cluster"] = self._max_per_cluster.value() or None
        if self._bundle_video is not None:
            options["include_video"] = self._bundle_video.isChecked()
            options["include_datasets"] = self._bundle_datasets.isChecked()
            options["gzip"] = self._bundle_gzip.isChecked()
        if self._kp_cutoff is not None:
            if self._kp_cutoff.value() > self._kp_cutoff.minimum():
                options["cutoff"] = self._kp_cutoff.value()
            options["blank_below_cutoff"] = self._kp_blank.isChecked()
            if self._kp_source is not None:
                options["source"] = self._kp_source.currentData()
            if self._kp_overlay is not None:
                options["overlay"] = self._kp_overlay.isChecked()
        if self._codec is not None:
            options["codec"] = self._codec.codec()
        # Only when asked for, as on the command line: light is every target's own default.
        if self._dark_figures is not None and self._dark_figures.isChecked():
            options["theme"] = "dark"
        # Last, so a control the user can actually see wins over a seed. The seed is for facts
        # no control offers (which track, which embedding run), not for overriding a choice.
        for key, value in self._seed.items():
            options.setdefault(key, value)
        return options

    def _export(self) -> None:
        parent = self._out.text().strip()
        if not parent:
            QMessageBox.warning(
                self, "Export", "Choose where to save the export first."
            )
            return
        chosen = self._chosen()
        if not chosen:
            QMessageBox.warning(self, "Export", "Tick at least one thing to export.")
            return
        options = self._options()
        if any(t.renders for t in chosen) and not options.get("renders"):
            QMessageBox.warning(
                self,
                "Export",
                "Choose at least one video to write (raw, overlay or mask), or untick Clips.",
            )
            return
        if (
            self._recording_list is not None
            and self._per_recording_chosen()
            and not self._ticked_recordings()
        ):
            QMessageBox.warning(self, "Export", "Tick at least one recording.")
            return
        if self._estimate is not None and self._estimate.fits is False:
            QMessageBox.warning(
                self,
                "Export",
                f"This export is estimated at {exporters.format_size(self._estimate.bytes)} "
                f"but only {exporters.format_size(self._estimate.free_bytes)} is free where "
                "you are saving it. Choose another folder, or untick some recordings or "
                "targets.",
            )
            return
        scope = self._final_scope()
        # Snapshot every Qt read above, on the UI thread — the worker must not touch a widget.
        subject = self._subject
        target_ids = [t.id for t in chosen]

        # **Submit a run; do not export here.** The dialog used to hold a `run_progress` overlay
        # for the whole export, which is exactly what #354 exists to stop: a 40-minute clip
        # export is not something to keep a modal window open for. What is long is the *run*,
        # and a run does not live in a dialog.
        #
        # Off the UI thread for a measured reason rather than ceremony: `run_export` calls
        # `import_runner_modules`, which is 0.79 s before it touches disk.
        self._set_busy(True)
        call = BackgroundCall(
            lambda: Experiment(subject.experiment_dir).run_export(
                subject.kind, subject.id, target_ids, parent, scope=scope, **options
            ),
            receiver=self,
        )
        self._submit_call = call  # keep it alive until it has emitted
        call.done.connect(self._on_submitted)
        call.failed.connect(self._on_submit_failed)
        call.start()

    def _set_busy(self, busy: bool) -> None:
        """Freeze the form in place while the run is submitted — no overlay.

        An overlay is for waiting on *work*; submitting is short, and the work has moved to a
        run with its own page. `launch._set_form_busy` is the same gesture over a different form
        and names that dialog's own fields, so it cannot be borrowed.
        """
        self._busy = busy
        self._buttons.setEnabled(not busy)
        self._sync_export_enabled()
        for check in (*self._include.values(), *self._renders.values()):
            check.setEnabled(not busy and check.isEnabled())
        self._out.setEnabled(not busy)

    def _on_submitted(self, handle) -> None:
        """Hand the handle up and close. The shell takes it from here.

        No "wrote N files" box: nothing has been written yet. What replaces it is the run's own
        page, which the shell opens with this handle live — the same thing launching a
        processing or analysis run already does.
        """
        self.run_handle = handle
        self.accept()

    def _on_submit_failed(self, exc: BaseException) -> None:
        """A refused export leaves the form open and usable, not a closed dialog and a beep.

        Submission is where a bad target or an unavailable one is caught (`run_export`
        validates before it reserves an id), so this is the path a user actually meets.
        """
        self._set_busy(False)
        QMessageBox.critical(self, "Export", str(exc))


def _mono():
    """The package's monospace font, for a path (GUI style: alignment matters)."""
    from PyQt6.QtGui import QFont

    return mono_font()


_RENDER_LABELS = {
    clip_render.RAW: "Raw clip",
    clip_render.OVERLAY: "Raw + mask overlay",
    clip_render.MASK: "Mask only",
}
