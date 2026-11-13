"""The window a quick preview opens into: one clip, whole frame, at its real rate.

**Why a popup rather than a bigger thumbnail.** A strip's preview is a compromise made for
*comparing* — a square crop of the subject, strided to a thumbnail's frame budget, sized so a
row of them fits side by side. Every one of those is wrong the moment somebody stops comparing
and starts looking. Double-click undoes all three at once: the whole frame instead of the crop,
every frame instead of a stride, and a window instead of a 132px square.

It shows the **whole frame**, not the crop that was clicked. The crop exists to let members be
told apart in a row; the reason to open one is almost always "what was actually happening
there", which is exactly what the crop removes — the arena, the walls, the other animal. The
subject stays marked, so which one it is never becomes a guess.

Built on `_ClipPlayer` rather than a new player: it already knows how to hold a clip with
surrounding context you can scrub into, at the recording's own frame rate, and reusing it is what
keeps this from being a fifth playback implementation.

**Advanced, by default; Simple a tick away.** Advanced is three pictures (``panels=True``,
``gui/clip_panes.py`` — the footage, the masks alone and the keypoints alone on white), the
strips under the slider (``gui/clip_strips.py``), and, where the opener can label a cluster, the
labelling row (``gui/clip_labelling.py``). Simple is the one-picture player every other surface
uses. Which one opens is remembered across windows and sessions, because it is a person's way of
working rather than a property of a clip.

**The picture is the point.** A person opens this window to decide what an animal is doing, so
the footage and its two panes take the window's whole width and every pixel of height the rest
does not need: one header row (the view ticks, the orientation, Export ▾, the clip's length and
place, previous / next), the picture with the stretch, then the slider, thin strips, the
player's controls and one labelling row — each only as tall as it has to be. The cluster
sections (what sets the cluster apart, its sibling) are one click away under **Cluster
details**, closed until asked for: they are for comparing, and open by default they took a third
of the width from the footage.

**The window's height follows its width** (:meth:`ClipPopup.content_height`). The picture keeps
its aspect ratio, so a window opened at a fixed height is, on a laptop, wider-limited: the picture
cannot use the height and the player centres it, leaving a band above it and another above
**Close**. The height is the header, the player at this width (``height_for_width``), the
labelling row and the buttons; the base (`Dialog.follow_content_height`) keeps the window at it
when it opens, when its width changes and whenever the player's shape changes. The window types
no size of its own (`a_dialog_never_types_its_opening_size`): the width is the base's fit to
its content.

**Blind mode** shows a recording as *recording A*, *B*, … in the title instead of its name: a
recording's name often carries its treatment or group, and judging a clip with that in view is
the bias blinding exists to remove. Also remembered.

**Non-modal, one per page** (11-9 meeting): every page opens it through :func:`open_clip_window`,
which shows the window beside the page rather than blocking it, and — while one is open for that
page — shows the next clip double-clicked **in the same window** (its previous / next sequence,
its cluster's labelling and lane replaced) and raises it, rather than stacking a second. It is
the page's child, so it closes with the page and with the app.

**As recorded ▾ / Aligned**: the orientation menu, remembered under :data:`ALIGNED_KEY`. Aligned
turns the footage — and the masks and keypoints drawn on it, and the side panes — so the body
points up, frame by frame (``ClipQueryPreview.clip_views``' ``aligned``: one
``api.clip_views.turn_matrix``, the montage previews' own turn). Greyed with the reason when the
clip's embedding has no keypoint run or no heading definition fits. **Align pose** in the
player's row is a different thing: the keypoints pane alone, drawn body-aligned at a fixed size.

**Export ▾**: *Video of what you see…* (an MP4, its codec chosen in the save dialog), *GIF of
what you see…* — the window's picture as it is on screen (the plain footage, or the footage with
its two panes; every tick; the orientation) over the whole span on the bar, context included —
and *This frame as PNG…*. Each runs off the UI thread (``api.clip_pane_export.export_clip_view``
/ ``write_clip_frame``), video and GIF behind a bar with Cancel. The player's own **Export
video…** is hidden here: it wrote the clip alone, and two exports in one window that wrote
different pictures would be a guess. An enlarged side pane exports itself
(``gui/clip_pane_window.py``).
"""

from PyQt6.QtCore import QEvent, QSettings, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from detail.gui.dialog import Dialog
from detail.gui.lifetime import cleared
from detail.gui.sizing import lines, set_available, text_width

#: The remembered choices (``QSettings``): Advanced (default on), Blind mode (default off), and
#: whether the cluster sections are open (default closed).
ADVANCED_KEY = "clip_window/advanced"
BLIND_KEY = "clip_window/blind"
DETAILS_KEY = "clip_window/details"
#: The orientation switch (As recorded / Aligned), remembered for the window.
ALIGNED_KEY = "interpret/clip_window_aligned"
#: What **Aligned** says while the clip's keypoints are being read.
ALIGN_CHECKING = "Reading this clip's keypoints…"
ALIGN_TIP = (
    "Turn the footage, and the masks and keypoints with it, so the animal's body points up, "
    'frame by frame. A frame with no body axis plays as recorded, marked "no axis".'
)


def remembered(key: str, default: bool) -> bool:
    return bool(QSettings().value(key, default, type=bool))


def remember(key: str, on: bool) -> None:
    QSettings().setValue(key, bool(on))


class ClipPopup(Dialog):
    """One clip, playing, in a window of its own."""

    def __init__(
        self,
        experiment,
        clip,
        parent=None,
        *,
        context_s: float,
        sequence=None,
        index=0,
        lane=None,
        order_note: str = "",
        labelling=None,
        advanced: bool | None = None,
    ) -> None:
        """``sequence`` (a list of clips, ``clip`` among them at ``index``) adds **◀ previous**
        / **next ▶** that swap the clip in the same player: stepping through every bout of one
        label, say, without closing and reopening a window per bout.

        ``context_s`` is the footage either side the popup opens with — **the opener's** call,
        not this class's, because the popup is shared: a montage opens it with its own context,
        a query grid on a match judged on its own. The viewer changes it from the player's row.

        ``lane`` (``lane(clip, start_ns, end_ns)`` → ``NodeView.lane`` segments) fills the
        cluster strip under the slider; only a clustering page has clusters to give, so the
        other openers pass none and the strip stays empty. ``order_note`` says how the
        sequence is ordered ("most typical first") beside the position.

        ``labelling`` (a ``clip_labelling.ClusterLabelling``) adds the labelling row in the
        Advanced view: label the cluster, and answer **Fits / Doesn't / Unclear** (Y / N / U)
        for each clip, which moves on to the next; and **Cluster details** beside it, which
        opens what sets the cluster apart and its sibling under the row.

        ``advanced`` opens in that view (``True``) or the plain one (``False``) whatever was
        last chosen. ``None`` opens the view last chosen. Either way, ticking the box later is
        remembered as usual.

        Pages open it through :func:`open_clip_window`, never by building it themselves.
        """
        super().__init__(parent)
        self._experiment = experiment
        self._context_s = float(context_s)
        self._lane = lane
        self._labelling = labelling
        self._clip = clip
        self._order_note = order_note
        self._sequence = list(sequence or [])
        self._index = int(index)
        self._aliases = _aliases([*self._sequence, clip])
        # `body()` / `set_buttons()` rather than a layout of this dialog's own: the shared base
        # owns the scroll and pins the button box outside it, so the way out never scrolls away.
        layout = self.body()

        # One header row: the view ticks, what the clip is, and stepping between clips — the
        # facts and controls that are not the picture, in the least height they fit.
        modes = QHBoxLayout()
        self._advanced = QCheckBox("Advanced view")
        self._advanced.setObjectName("clip_advanced")
        self._advanced.setChecked(
            remembered(ADVANCED_KEY, True) if advanced is None else bool(advanced)
        )
        self._advanced.setToolTip(
            "The footage with its masks and keypoints beside it, the strips under the slider "
            "and, from a cluster, labelling. Untick for the plain player."
        )
        self._advanced.toggled.connect(self._on_advanced)
        modes.addWidget(self._advanced)
        self._blind = QCheckBox("Blind mode")
        self._blind.setObjectName("clip_blind")
        self._blind.setChecked(remembered(BLIND_KEY, False))
        self._blind.setToolTip(
            "Show recordings as A, B, C instead of their names, which can give away a "
            "treatment or group."
        )
        self._blind.toggled.connect(self._on_blind)
        modes.addWidget(self._blind)
        modes.addSpacing(text_width("MM"))
        # A menu, as on the montages: the header row has the clip's facts to fit as well.
        self._orientation = QComboBox()
        self._orientation.setObjectName("clip_orientation")
        self._orientation.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._orientation.addItem("As recorded", False)
        self._orientation.addItem("Aligned", True)
        self._orientation.activated.connect(
            lambda i: self.set_aligned(bool(self._orientation.itemData(i)))
        )
        modes.addWidget(self._orientation)
        self._wants_aligned = remembered(ALIGNED_KEY, False)
        self._align_note: str | None = ALIGN_CHECKING
        self._align_generation = 0
        self._export_menu = QMenu(self)
        self._export_menu.setToolTipsVisible(True)
        self._export_menu.addAction("Video of what you see…", self.export_video)
        self._export_menu.addAction("GIF of what you see…", self.export_gif)
        self._export_menu.addSeparator()
        self._export_menu.addAction("This frame as PNG…", self.export_png)
        self._export = QPushButton("Export ▾")
        self._export.setObjectName("clip_export")
        self._export.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._export.setToolTip(
            "Write what this window shows: as a video or a GIF over the whole span on the bar "
            "(the clip and its context), or the frame on screen as a picture."
        )
        self._export.setMenu(self._export_menu)
        modes.addSpacing(text_width("M"))
        modes.addWidget(self._export)
        modes.addSpacing(text_width("MM"))
        # The clip's length and where it sits in its recording: the two facts about a clip the
        # picture cannot show, and the first a person asks when two clips look alike. Never
        # widens the window: the whole line is on hover.
        self._about = QLabel("")
        self._about.setObjectName("clip_about")
        self._about.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        modes.addWidget(self._about, 1)
        # Built whether or not there is a sequence: a reused window may be handed one later.
        self._previous = QPushButton("◀ previous")
        self._previous.clicked.connect(lambda: self.step(-1))
        modes.addWidget(self._previous)
        self._position = QLabel("")
        modes.addWidget(self._position)
        self._next = QPushButton("next ▶")
        self._next.clicked.connect(lambda: self.step(1))
        modes.addWidget(self._next)
        layout.addLayout(modes)
        #: What the last export wrote.
        self._written = QLabel("")
        self._written.setVisible(False)
        layout.addWidget(self._written)

        self._row = QVBoxLayout()
        self._row.setSpacing(lines(1) // 4)
        layout.addLayout(self._row, 1)
        self._player = None
        self._panel = self._fingerprint = self._compare = None
        self._labelling_row = self._details = self._details_button = None
        self._build_view(self._advanced.isChecked())
        self._show_clip()
        self._sync_orientation()
        self._read_alignment()

        for _text, key, verdict in _answer_keys():
            QShortcut(
                QKeySequence(key), self, activated=lambda v=verdict: self._answer(v)
            )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)

    # -- building the two views ------------------------------------------------ #
    def _build_view(self, advanced: bool) -> None:
        from detail.gui.behavior_query import _ClipPlayer  # local: import cycle

        if self._player is not None:
            self._context_s = float(self._player.context_s())
            self._row.removeWidget(self._player)
            self._player.deleteLater()
        self._drop_labelling()
        # Context used to be off here for want of a way to say, on the bar, which part was the
        # behaviour — a padded window with nothing marking the clip is worse than a shorter one.
        # The player now draws the clip as a band on its own bar, so the padding is labelled.
        self._player = _ClipPlayer(
            self._experiment, autoplay=True, context_s=self._context_s, panels=advanced
        )
        # The picture's stretch: everything else in the window takes only what it needs.
        self._row.insertWidget(0, self._player, 1)
        # The player posts a layout request when its shape changes — a frame of a new aspect,
        # a control row that wraps — and that is when the window's height is re-asked.
        self._player.installEventFilter(self)
        # The window's Export ▾ writes what is on screen; the player's own wrote the clip alone.
        self._player.export_button().setVisible(False)
        self._player.set_aligned(self.is_aligned())
        if advanced and self._lane is not None:
            self._player.set_lane_source(self._lane)
        if advanced and self._labelling is not None:
            self._build_labelling()
        self._player.set_clip(self._clip)
        self.follow_content_height()

    def _drop_labelling(self) -> None:
        """Take out the labelling row and the cluster sections (a view change, a new cluster)."""
        for widget in (self._labelling_row, self._details):
            if widget is not None:
                self._row.removeWidget(widget)
                widget.deleteLater()
        self._panel = self._fingerprint = self._compare = None
        self._labelling_row = self._details = self._details_button = None

    def _set_cluster(self, labelling, lane) -> None:
        """A reused window's new opener context: the cluster to label and its lane. The same
        cluster keeps the labelling it has — its checks queued in order, its tracks read.
        """
        same = (
            labelling is not None
            and self._labelling is not None
            and type(labelling) is type(self._labelling)
            and labelling.node() == self._labelling.node()
        )
        self._lane = lane
        if self.is_advanced():
            self._player.set_lane_source(lane)
        if same:
            return
        self._drop_labelling()
        self._labelling = labelling
        if self.is_advanced() and labelling is not None:
            self._build_labelling()
        self.follow_content_height()

    def _build_labelling(self) -> None:
        """The labelling row under the player, and the closed cluster sections under it."""
        from detail.gui.clip_labelling import ClipLabelPanel

        self._labelling_row = QWidget()
        self._labelling_row.setObjectName("clip_labelling_row")
        row = QHBoxLayout(self._labelling_row)
        row.setContentsMargins(0, 0, 0, 0)
        self._panel = ClipLabelPanel(self._labelling)
        self._panel.answered.connect(self._after_answer)
        row.addWidget(self._panel, 1)
        self._details_button = QPushButton("Cluster details")
        self._details_button.setObjectName("clip_details")
        self._details_button.setCheckable(True)
        self._details_button.setToolTip(
            "What sets this cluster apart, with the clip on screen marked, and a clip of its "
            "sibling in the tree to compare with. Takes room from the picture while open."
        )
        self._details_button.toggled.connect(self.show_details)
        row.addWidget(self._details_button)
        self._row.addWidget(self._labelling_row)
        self._panel.set_clip(self._clip)
        if remembered(DETAILS_KEY, False):
            self._details_button.setChecked(True)

    def show_details(self, on: bool) -> None:
        """Open or close the cluster sections, building them the first time they open: they
        measure the cluster and loop a sibling clip, which nobody should pay for unasked.
        """
        remember(DETAILS_KEY, on)
        if self._details_button is not None and self._details_button.isChecked() != on:
            self._details_button.setChecked(on)  # re-enters with the same `on`
            return
        if on and self._details is None and self._labelling is not None:
            from detail.gui.clip_cluster_info import (
                ClipComparePanel,
                ClipFingerprintPanel,
            )

            self._details = QWidget()
            self._details.setObjectName("clip_details_sections")
            sections = QHBoxLayout(self._details)
            sections.setContentsMargins(0, 0, 0, 0)
            self._fingerprint = ClipFingerprintPanel(self._labelling, self._clip)
            sections.addWidget(self._fingerprint, 1)
            self._compare = ClipComparePanel(self._labelling)
            self._compare.open_clip.connect(self.jump_to)
            sections.addWidget(self._compare, 1)
            self._row.addWidget(self._details)
        if self._details is not None:
            self._details.setVisible(on)
            if self._compare is not None and self._compare.sibling_player() is not None:
                player = self._compare.sibling_player()
                player.play() if on else player.pause()
        self.follow_content_height()

    def details_shown(self) -> bool:
        return self._details is not None and self._details.isVisibleTo(self)

    def _on_advanced(self, on: bool) -> None:
        remember(ADVANCED_KEY, on)
        self._build_view(on)
        self._player.focus_scrubber()

    def _on_blind(self, on: bool) -> None:
        remember(BLIND_KEY, on)
        self._show_clip()

    def is_advanced(self) -> bool:
        return self._advanced.isChecked()

    def label_panel(self):
        """The labelling row's panel, or ``None`` (Simple view, or an opener with no cluster)."""
        return self._panel

    def fingerprint_panel(self):
        """What sets the cluster apart — ``None`` until **Cluster details** is first opened."""
        return self._fingerprint

    def compare_panel(self):
        """The sibling and the grid — ``None`` until **Cluster details** is first opened."""
        return self._compare

    # -- the orientation ------------------------------------------------------- #
    def is_aligned(self) -> bool:
        """Whether the picture is drawn **Aligned**: chosen, and this clip's run can."""
        return self._wants_aligned and self._align_note is None

    def alignment_note(self) -> str | None:
        return self._align_note

    def set_aligned(self, on: bool) -> None:
        """Choose the orientation (remembered); drawn Aligned only while the clip can be."""
        self._wants_aligned = bool(on)
        remember(ALIGNED_KEY, self._wants_aligned)
        self._sync_orientation()

    def set_alignment(self, note: str | None) -> None:
        """Whether this clip can be drawn Aligned: ``None`` it can, else why not."""
        self._align_note = note
        self._sync_orientation()

    def _sync_orientation(self) -> None:
        self._orientation.blockSignals(True)
        self._orientation.setCurrentIndex(1 if self.is_aligned() else 0)
        self._orientation.blockSignals(False)
        set_available(
            self._orientation, self._align_note is None, self._align_note or ""
        )
        if self._align_note is None:
            self._orientation.setToolTip(ALIGN_TIP)
        if self._player is not None:
            self._player.set_aligned(self.is_aligned())

    def _read_alignment(self) -> None:
        """Ask, off the UI thread, whether this clip's run can align it (its keypoint run and
        heading definition); a later clip supersedes a read in flight."""
        from detail.gui.workers import BackgroundCall

        self._align_generation += 1
        generation = self._align_generation
        clip, experiment_dir = self._clip, self._experiment.dir
        self.set_alignment(ALIGN_CHECKING)

        def read():
            from detail.api.clip_views import alignment_note

            return alignment_note(  # off-thread (BackgroundCall)
                experiment_dir, {clip.recording_id: clip.embed_run_id}
            )

        call = BackgroundCall(read, receiver=self)
        call.done.connect(
            lambda note: generation == self._align_generation
            and self.set_alignment(note)
        )
        call.failed.connect(
            lambda exc: generation == self._align_generation
            and self.set_alignment(f"This clip's keypoints could not be read: {exc}")
        )
        call.start()

    # -- showing another clip in the same window ------------------------------- #
    def show_clip(
        self,
        clip,
        *,
        sequence=None,
        index: int = 0,
        order_note: str = "",
        labelling=None,
        lane=None,
    ) -> None:
        """Show ``clip`` in this window, with a new previous / next ``sequence`` and the
        opener's ``labelling`` / ``lane`` for it — what a page's second double-click does while
        the window is open (:func:`open_clip_window`)."""
        self._sequence = list(sequence or [])
        self._index = int(index)
        self._order_note = order_note
        self._aliases = _aliases([*self._sequence, clip])
        self._clip = clip
        self._set_cluster(labelling, lane)
        self._show(clip)

    # -- exporting what is on screen ------------------------------------------- #
    def export_settings(self):
        """``(settings, panes)``: what the picture is drawn with now
        (``api.clip_pane_export.ClipViewSettings``) and whether it is the footage with its two
        panes (Advanced) or the footage alone."""
        return self._player.view_settings(), self.is_advanced()

    def export_frames(self) -> tuple[int, int]:
        """The inclusive global frames on the bar — the clip and its context."""
        return self._player.window()

    def _suggested(self, kind: str) -> str:
        from detail.gui.animation_export import safe_name

        return f"{safe_name(self._clip.label or 'clip')}.{kind}"

    def export_video(self) -> None:
        from detail.api.animation import MP4
        from detail.gui.animation_export import ask_mp4_path

        answer = ask_mp4_path(self, "Export video", self._suggested(MP4))
        if answer is not None:
            path, codec = answer
            self.write_view(path, codec=codec)

    def export_gif(self) -> None:
        from detail.api.animation import GIF
        from detail.gui.animation_export import ask_gif_settings, ask_save_path

        first, last = self.export_frames()
        answer = ask_gif_settings(
            self,
            f"What this window shows, over the {last - first + 1:,} frames on the bar "
            "(the clip and its context).",
        )
        if answer is None:
            return
        path = ask_save_path(self, "Export GIF", self._suggested(GIF), GIF)
        if path is not None:
            self.write_view(path, max_width=answer[0], fps=answer[1])

    def write_view(
        self, path, *, max_width: int | None = None, fps=None, codec: str | None = None
    ):
        """Write what the window shows over the frames on the bar to ``path``, off the UI
        thread behind a bar with Cancel; an mp4 encoded as ``codec`` (the default when
        ``None``)."""
        from detail.api import clip_pane_export
        from detail.core.video_codecs import DEFAULT_VIDEO_CODEC
        from detail.gui.animation_export import run_export, written_note
        from detail.gui.video_codec import with_codec_note

        codec = codec or DEFAULT_VIDEO_CODEC
        clip, (first, last) = self._clip, self.export_frames()
        settings, panes = self.export_settings()
        experiment_dir = self._experiment.dir

        def work(progress):
            return clip_pane_export.export_clip_view(  # off-thread (run_progress)
                experiment_dir,
                recording_id=clip.recording_id,
                embed_run_id=clip.embed_run_id,
                track_id=clip.track_id,
                first=first,
                last=last,
                settings=settings,
                panes=panes,
                path=path,
                max_width=max_width,
                fps=fps,
                progress=progress,
                codec=codec,
            )

        return run_export(
            self,
            work,
            on_written=lambda written: self._show_written(
                with_codec_note(written_note(written), written, codec)
            ),
            label="Exporting what you see…",
        )

    def export_png(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getSaveFileName(
            self, "Export this frame", self._suggested("png"), "PNG picture (*.png)"
        )
        if path:
            self.write_png(path)

    def write_png(self, path):
        """Write the frame under the playhead as the window shows it, off the UI thread."""
        from detail.api import clip_pane_export
        from detail.gui.animation_export import written_note
        from detail.gui.busy import run_busy

        clip, frame = self._clip, self._player.current_frame()
        settings, panes = self.export_settings()
        experiment_dir = self._experiment.dir

        def work():
            return clip_pane_export.write_clip_frame(  # off-thread (run_busy)
                experiment_dir,
                recording_id=clip.recording_id,
                embed_run_id=clip.embed_run_id,
                track_id=clip.track_id,
                frame=frame,
                settings=settings,
                panes=panes,
                path=path,
            )

        return run_busy(
            self,
            work,
            lambda written: self._show_written(written_note(written)),
            label="Writing the frame…",
        )

    def _show_written(self, text: str) -> None:
        self._written.setText(text)
        self._written.setVisible(True)

    def export_menu(self) -> QMenu:
        return self._export_menu

    # -- moving through the clips --------------------------------------------- #
    def step(self, by: int) -> None:
        """Show the clip ``by`` places along the sequence (clamped to its ends)."""
        if not self._sequence:
            return
        self._index = min(max(self._index + by, 0), len(self._sequence) - 1)
        self._show(self._sequence[self._index])

    def jump_to(self, clip) -> None:
        """Show ``clip``: its place in the sequence when it has one, else on its own."""
        for index, other in enumerate(self._sequence):
            if (other.recording_id, other.track_id, other.start_t_ns) == (
                clip.recording_id,
                clip.track_id,
                clip.start_t_ns,
            ):
                self._index = index
                self._show(other)
                return
        self._show(clip)

    def _show(self, clip) -> None:
        self._clip = clip
        self._player.set_clip(clip)
        if self._panel is not None:
            self._panel.set_clip(clip)
        if self._fingerprint is not None:
            self._fingerprint.set_clip(clip)
        self._show_clip()
        self._read_alignment()

    def _answer(self, verdict: str) -> None:
        if self._panel is not None:
            self._panel.answer(verdict)

    def _after_answer(self, _verdict: str) -> None:
        if self._sequence and self._index < len(self._sequence) - 1:
            self.step(1)

    @property
    def index(self) -> int:
        return self._index

    # -- what the window says about the clip ----------------------------------- #
    def _show_clip(self) -> None:
        from detail.gui.timefmt import format_time

        clip = self._clip
        if self._blind.isChecked():
            title = (
                f"Clip {self._index + 1} · recording {self._aliases[clip.recording_id]}"
            )
        else:
            title = clip.label or "Clip"
        self.setWindowTitle(title)
        length_s = max(int(clip.end_t_ns) - int(clip.start_t_ns), 0) / 1e9
        self._about.setText(
            f"{title} · length {length_s:.1f} s · starts "
            f"{format_time(int(clip.start_t_ns))} into the recording"
        )
        self._about.setToolTip(self._about.text())
        stepping = len(self._sequence) > 1
        for widget in (self._previous, self._position, self._next):
            widget.setVisible(stepping)
        if stepping:
            note = f" ({self._order_note})" if self._order_note else ""
            self._position.setText(f"{self._index + 1} of {len(self._sequence)}{note}")
            set_available(self._previous, self._index > 0, "This is the first clip.")
            set_available(
                self._next,
                self._index < len(self._sequence) - 1,
                "This is the last clip.",
            )

    def player(self):
        """The player inside, for a test that wants to ask what it is doing."""
        return self._player

    def content_height(self) -> int | None:
        """The window as tall as its header, the player at the window's width, the labelling
        row and the buttons — no taller, so no band opens above the picture or above **Close**;
        no shorter, so nothing scrolls. The base caps it to the screen.

        The width stays where it is: it is the side the picture is limited by on a laptop, and
        the side a person sizes the window by.
        """
        player = self._player
        if player is None:
            return None
        self.layout().activate()
        chrome = self.height() - player.height()
        return chrome + player.height_for_width(player.width())

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 — Qt's name
        if cleared(self):
            return False
        # `getattr`: the base filters the body from before the player exists.
        player = getattr(self, "_player", None)
        if (
            player is not None
            and watched is player
            and event.type() == QEvent.Type.LayoutRequest
        ):
            self.follow_content_height()
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """Give the scrubber the keyboard, so space works without clicking first.

        A window opened to watch one clip has exactly one thing the keyboard could mean.
        """
        super().showEvent(event)
        self._player.focus_scrubber()


def open_clip_window(
    opener,
    experiment,
    clip,
    *,
    context_s: float,
    sequence=None,
    index: int = 0,
    order_note: str = "",
    advanced: bool | None = None,
    labelling=None,
    lane=None,
) -> ClipPopup:
    """Open ``clip`` in ``opener``'s clip window, **non-modally**: the page stays usable beside
    it. While that window is open, the clip is shown in it (with the new ``sequence``,
    ``labelling`` and ``lane``) and the window is raised — one per page, never a stack.
    Parented to ``opener``, so it closes with the page. Returns the window.

    ``context_s`` and ``advanced`` are the opener's, as for :class:`ClipPopup`; a reused window
    keeps the context and view its viewer last chose. ``labelling`` (a
    ``clip_labelling.ClusterLabelling``) and ``lane`` (``NodeView.lane`` around a clip) are a
    clustering page's: the labelling row, the clip checks, **Cluster details** and the cluster
    strip. Every other opener passes neither."""
    from PyQt6 import sip

    existing = getattr(opener, "_clip_window", None)
    if existing is not None and not sip.isdeleted(existing) and existing.isVisible():
        existing.show_clip(
            clip,
            sequence=sequence,
            index=index,
            order_note=order_note,
            labelling=labelling,
            lane=lane,
        )
        existing.raise_()
        existing.activateWindow()
        return existing
    window = ClipPopup(
        experiment,
        clip,
        opener,
        context_s=context_s,
        sequence=sequence,
        index=index,
        order_note=order_note,
        labelling=labelling,
        lane=lane,
        advanced=advanced,
    )
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    window.setModal(False)
    opener._clip_window = window
    window.show()
    return window


def _aliases(clips) -> dict[str, str]:
    """``{recording_id: "A"}``, lettered in the order the recordings first appear — the one
    lettering the clip window and the clip montage share (``api.clip_montage``)."""
    from detail.api.clip_montage import blind_aliases

    return blind_aliases([clip.recording_id for clip in clips])


def _answer_keys():
    from detail.gui.clip_labelling import ANSWERS

    return ANSWERS
