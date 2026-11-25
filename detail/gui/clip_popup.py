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

**Advanced, by default; Simple a tick away.** Advanced is ``gui/advanced_player.py``: three
pictures (the footage, the masks alone and the keypoints alone on white, ``gui/clip_panes.py``),
the strips under the slider (``gui/clip_strips.py``), and the view ticks — Keypoints, Trail and
which points, Mask trail, Align pose, Loop clip. Simple is the one-picture player every other
surface uses. Which one opens is remembered across windows and sessions, because it is a
person's way of working rather than a property of a clip.

**The picture is the point.** A person opens this window to decide what an animal is doing, so
the footage and its two panes take the window's whole width and every pixel of height the rest
does not need: one header row (the view ticks, the clip's length and place, previous / next),
the picture with the stretch, then the slider, thin strips and the player's controls — each
only as tall as it has to be.

**The window's height follows its width** (:meth:`ClipPopup.fit_height`). The picture keeps its
aspect ratio, so a window opened at a fixed height is, on a laptop, wider-limited: the picture
cannot use the height and the player centres it, leaving a band above it and another above
**Close** — measured at 1324x805 (a 1440x900 screen), 73px over the picture and ~60px under the
controls. The height is the header, the player at this width (``height_for_width``) and the
buttons, re-asked whenever the player's shape changes (a decoded frame's aspect, controls that
wrap, Simple / Advanced).

**Blind mode** shows a recording as *recording A*, *B*, … in the title instead of its name: a
recording's name often carries its treatment or group, and judging a clip with that in view is
the bias blinding exists to remove. Also remembered.

**Non-modal, one per page** (11-9 meeting): every page opens it through :func:`open_clip_window`,
which shows the window beside the page rather than blocking it, and — while one is open for that
page — shows the next clip double-clicked **in the same window** (its previous / next sequence
replaced) and raises it, rather than stacking a second. It is the page's child, so it closes with
the page and with the app.

**As recorded ▾ / Aligned**: the orientation menu, remembered under :data:`ALIGNED_KEY`. Aligned
turns the footage — and the masks and keypoints drawn on it, and the side panes — so the body
points up, frame by frame (``api.clip_views.turn_views``, the montage's own turn). Greyed with the
reason when the clip's embedding has no keypoint run or no heading definition fits.

**Export ▾**: *Video of what you see…*, *GIF of what you see…* — the window's picture as it is on
screen (the plain footage, or the footage with its two panes; every tick; the orientation) over the
whole span on the bar, context included — and *This frame as PNG…*. Each runs off the UI thread
(``api.clip_views.export_clip_view`` / ``write_clip_frame``), video and GIF behind a bar with
Cancel. The plain player's own **Export video…** is hidden here: it wrote the clip alone, raw or
overlaid, and two exports in one window that wrote different pictures would be a guess.
"""

from PyQt6.QtCore import QEvent, QSettings, QSize, Qt, QTimer
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
)

from detail.gui.dialog import Dialog
from detail.gui.sizing import lines, set_available, text_width

#: Opening size, in lines rather than pixels, so it follows a user who has asked for larger text.
#: The Advanced picture is 1.5 frames wide (the footage and its half-width side column), so at
#: 16px a line a 16:9 frame at this width is ~590px tall and the height leaves it that after the
#: rows under it. Capped to the screen (:func:`opening_size`), and resizable from here.
_WIDE = 100
_TALL = 56
#: The most of the screen's free height the window opens on.
_SCREEN_SHARE = 0.92
#: The most of the screen's free **width**: nearly all of it. The Advanced controls row is one
#: line of ~1340px at the system font, and at 92% of a 1440px laptop screen (1324px) the frame
#: counter wrapped onto a line of its own. The window is a viewer, wider-limited on a laptop, so
#: the width is what the picture and that row are short of.
_SCREEN_WIDTH_SHARE = 0.98
#: The remembered choices (``QSettings``): Advanced (default on) and Blind mode (default off).
ADVANCED_KEY = "clip_window/advanced"
BLIND_KEY = "clip_window/blind"
#: The orientation switch (As recorded / Aligned), remembered for the window.
ALIGNED_KEY = "interpret/clip_window_aligned"
#: What **Aligned** says while the clip's keypoints are being read.
ALIGN_CHECKING = "Reading this clip's keypoints…"
ALIGN_TIP = (
    "Turn the footage, and the masks and keypoints with it, so the animal's body points up, "
    "frame by frame. A side-view embedding is flipped where it flipped instead, never turned. "
    'A frame with no body axis plays as recorded, marked "no axis".'
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
        order_note: str = "",
        advanced: bool | None = None,
    ) -> None:
        """``sequence`` (a list of clips, ``clip`` among them at ``index``) adds **◀ previous**
        / **next ▶** that swap the clip in the same player: stepping through every bout of one
        label, say, without closing and reopening a window per bout. ``order_note`` says how
        the sequence is ordered ("most typical first") beside the position.

        ``context_s`` is the footage either side the popup opens with — **the opener's** call,
        not this class's, because the popup is shared: the clustering page opens it to tell two
        similar clusters apart, where a second either side is the point, and the query grids open
        it on a match being judged on its own. The viewer changes it from the player's row.

        ``advanced`` opens in that view (``True``) or the plain one (``False``) whatever was
        last chosen — the cluster tree's *Preview clips…* asks for Advanced by name. ``None``
        opens the view last chosen. Either way, ticking the box later is remembered as usual.
        """
        super().__init__(parent)
        self._experiment = experiment
        self._context_s = float(context_s)
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
            "The footage with its masks and keypoints beside it, trails, and the strips under "
            "the slider. Untick for the plain player."
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
        # Owned by the window, so a shot pending when it closes goes with it.
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self.fit_height)
        self._build_view(self._advanced.isChecked())
        self._show_clip()
        self._sync_orientation()
        self._read_alignment()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)
        self.resize(opening_size(self.screen()))

    # -- building the two views ------------------------------------------------ #
    def _build_view(self, advanced: bool) -> None:
        # Local imports: `behavior_query` imports this module's openers' neighbours (a cycle),
        # and the Advanced player is only paid for when asked.
        if self._player is not None:
            self._context_s = float(self._player.context_s())
            self._row.removeWidget(self._player)
            self._player.deleteLater()
        if advanced:
            from detail.gui.advanced_player import AdvancedClipPlayer as player_class
        else:
            from detail.gui.advanced_player import PlainClipPlayer as player_class
        # Context used to be off here for want of a way to say, on the bar, which part was the
        # behaviour — a padded window with nothing marking the clip is worse than a shorter one.
        # The player now draws the clip as a band on its own bar, so the padding is labelled.
        self._player = player_class(
            self._experiment, autoplay=True, context_s=self._context_s
        )
        # The picture's stretch: everything else in the window takes only what it needs.
        self._row.insertWidget(0, self._player, 1)
        # The player posts a layout request when its shape changes — a frame of a new aspect,
        # a control row that wraps — and that is when the window's height is re-asked.
        self._player.installEventFilter(self)
        # The window's Export ▾ writes what is on screen; the player's own wrote the clip alone.
        self._player.export_button().setVisible(False)
        self._player.set_aligned(self.is_aligned())
        self._player.set_clip(self._clip)
        if self.isVisible():
            self._fit_timer.start(0)

    def _on_advanced(self, on: bool) -> None:
        remember(ADVANCED_KEY, on)
        self._build_view(on)
        self._player.focus_scrubber()

    def _on_blind(self, on: bool) -> None:
        remember(BLIND_KEY, on)
        self._show_clip()

    def is_advanced(self) -> bool:
        return self._advanced.isChecked()

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
        self, clip, *, sequence=None, index: int = 0, order_note: str = ""
    ) -> None:
        """Show ``clip`` in this window, with a new previous / next ``sequence`` — what a
        page's second double-click does while the window is open (:func:`open_clip_window`).
        """
        self._clip = clip
        self._sequence = list(sequence or [])
        self._index = int(index)
        self._order_note = order_note
        self._aliases = _aliases([*self._sequence, clip])
        self._player.set_clip(clip)
        self._show_clip()
        self._read_alignment()

    # -- exporting what is on screen ------------------------------------------- #
    def export_settings(self):
        """``(settings, panes)``: what the picture is drawn with now and whether it is the
        footage with its two panes (Advanced) or the footage alone."""
        return self._player.view_settings(), self.is_advanced()

    def export_frames(self) -> tuple[int, int]:
        """The inclusive global frames on the bar — the clip and its context."""
        return self._player.window()

    def _suggested(self, kind: str) -> str:
        from detail.gui.animation_export import safe_name

        return f"{safe_name(self._clip.label or 'clip')}.{kind}"

    def export_video(self) -> None:
        from detail.api.animation import MP4
        from detail.gui.animation_export import ask_save_path

        path = ask_save_path(self, "Export video", self._suggested(MP4), MP4)
        if path is not None:
            self.write_view(path)

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

    def write_view(self, path, *, max_width: int | None = None, fps=None):
        """Write what the window shows over the frames on the bar to ``path``, off the UI
        thread behind a bar with Cancel."""
        from detail.api import clip_views
        from detail.gui.animation_export import run_export

        clip, (first, last) = self._clip, self.export_frames()
        settings, panes = self.export_settings()
        experiment_dir = self._experiment.dir

        def work(progress):
            return clip_views.export_clip_view(  # off-thread (run_progress)
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
            )

        return run_export(
            self, work, on_written=self._show_written, label="Exporting what you see…"
        )

    def export_png(self) -> None:
        from detail.gui.animation_export import ask_save_path

        path = ask_save_path(
            self,
            "Export this frame",
            self._suggested("png"),
            "png",
            "PNG picture (*.png)",
        )
        if path is not None:
            self.write_png(path)

    def write_png(self, path):
        """Write the frame under the playhead as the window shows it, off the UI thread."""
        from detail.api import clip_views
        from detail.gui.busy import run_busy

        clip, frame = self._clip, self._player.current_frame()
        settings, panes = self.export_settings()
        experiment_dir = self._experiment.dir

        def work():
            return clip_views.write_clip_frame(  # off-thread (run_busy)
                experiment_dir,
                recording_id=clip.recording_id,
                embed_run_id=clip.embed_run_id,
                track_id=clip.track_id,
                frame=frame,
                settings=settings,
                panes=panes,
                path=path,
            )

        return run_busy(self, work, self._show_written, label="Writing the frame…")

    def _show_written(self, path) -> None:
        from detail.gui.animation_export import written_note

        self._written.setText(written_note(path))
        self._written.setVisible(True)

    def export_menu(self) -> QMenu:
        return self._export_menu

    # -- moving through the clips --------------------------------------------- #
    def step(self, by: int) -> None:
        """Show the clip ``by`` places along the sequence (clamped to its ends)."""
        if not self._sequence:
            return
        self._index = min(max(self._index + by, 0), len(self._sequence) - 1)
        self._clip = self._sequence[self._index]
        self._player.set_clip(self._clip)
        self._show_clip()
        self._read_alignment()

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

    def fit_height(self, max_height: int | None = None) -> None:
        """Make the window as tall as its header, the player at the window's width and the
        buttons — no taller, so no band opens above the picture or above **Close**; no shorter,
        so nothing scrolls. Capped at ``max_height`` (default: the screen's share).

        The width stays where it is: it is the side the picture is limited by on a laptop, and
        the side a person sizes the window by.
        """
        player = self._player
        if player is None or not self.isVisible():
            return
        if max_height is None:
            screen = self.screen()
            max_height = (
                int(screen.availableGeometry().height() * _SCREEN_SHARE)
                if screen is not None
                else self.height()
            )
        self.layout().activate()
        chrome = self.height() - player.height()
        wanted = chrome + player.height_for_width(player.width())
        height = max(min(wanted, max_height), self.minimumSizeHint().height())
        if height != self.height():
            self.resize(self.width(), height)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 — Qt's name
        # `getattr`: the base filters the body from before the player exists.
        player = getattr(self, "_player", None)
        if (
            player is not None
            and watched is player
            and event.type() == QEvent.Type.LayoutRequest
        ):
            self._fit_timer.start(0)
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """A new width is a new picture height: fit again. Only on a width change — the fit
        itself changes the height alone, so it never sets itself off again."""
        super().resizeEvent(event)
        timer = getattr(
            self, "_fit_timer", None
        )  # a resize may come before it is built
        if timer is not None and event.oldSize().width() != event.size().width():
            timer.start(0)

    def showEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """Give the scrubber the keyboard, so space works without clicking first.

        A window opened to watch one clip has exactly one thing the keyboard could mean.
        """
        super().showEvent(event)
        self._player.focus_scrubber()
        self._fit_timer.start(0)


def opening_size(screen) -> QSize:
    """The window's opening size: ``_WIDE`` x ``_TALL`` lines, capped to ``screen``'s free
    area — nearly all of its width, ``_SCREEN_SHARE`` of its height. :meth:`ClipPopup.fit_height`
    then trims the height to what the picture at that width needs."""
    size = QSize(lines(_WIDE), lines(_TALL))
    if screen is None:
        return size
    free = screen.availableGeometry()
    return size.boundedTo(
        QSize(
            int(free.width() * _SCREEN_WIDTH_SHARE),
            int(free.height() * _SCREEN_SHARE),
        )
    )


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
) -> ClipPopup:
    """Open ``clip`` in ``opener``'s clip window, **non-modally**: the page stays usable beside
    it. While that window is open, the clip is shown in it (with the new ``sequence``) and the
    window is raised — one per page, never a stack. Parented to ``opener``, so it closes with
    the page. Returns the window.

    ``context_s`` and ``advanced`` are the opener's, as for :class:`ClipPopup`; a reused window
    keeps the context and view its viewer last chose."""
    from PyQt6 import sip

    existing = getattr(opener, "_clip_window", None)
    if existing is not None and not sip.isdeleted(existing) and existing.isVisible():
        existing.show_clip(clip, sequence=sequence, index=index, order_note=order_note)
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
        advanced=advanced,
    )
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    window.setModal(False)
    opener._clip_window = window
    window.show()
    return window


def _aliases(clips) -> dict[str, str]:
    """``{recording_id: "A"}``, lettered in the order the recordings first appear — the one
    lettering the clip window and the clip montage share (``api.clip_views.blind_aliases``).
    """
    from detail.api.clip_views import blind_aliases

    return blind_aliases([clip.recording_id for clip in clips])
