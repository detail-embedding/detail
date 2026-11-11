"""One of the clip window's side panes, large: the masks alone or the keypoints alone, at the
footage's full resolution, with its own playback and export (§22 *Enlarged panes*).

Opened by clicking the Masks or Keypoints pane in the clip window's Advanced view
(``ClipPanes.enlarge_wanted`` → ``AdvancedClipPlayer.open_pane``). The pane beside the footage is
a half-height thumbnail; this is the same picture — ``api.clip_views``'s ``masks`` or ``pose``,
not the thumbnail scaled up — in a window that can be made as large as the screen.

**It follows the clip window, live.** It reads the player's view ticks (Mask trail, Trail and Trail
points, Align pose) and its window of frames (Context) whenever they change, and the clip when
**previous / next** moves on. So it is opened non-modally, beside the clip window, and is that
window's child: it closes with it. Its own playback — play and pause, stepping, the slider, speed,
loop — starts at the frame the clip window was on and is independent from there.

**One window per pane, and it dies when closed.** Clicking a pane whose window is open raises
that window rather than opening a second; Close deletes the window
(``WA_DeleteOnClose``), stops its playback and unhooks it from the player. The port this
replaced built a new window on every click and set none of that, so a closed window stayed
parented to the main window, still connected to the player, decoding frames in a loop forever.

**Export video…** writes this pane over the frames its bar spans as an mp4, every frame at full
size; **Export GIF…** asks a width and a frame rate first. Both cover the clip, plus its context
when the clip window's Context is on — the frames on this window's bar, which the GIF dialog
states.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QCheckBox, QDialogButtonBox, QLabel, QPushButton

from detail.api import clip_views
from detail.api.animation import GIF, MP4
from detail.gui import frame_source
from detail.gui.animation_export import (
    ask_gif_settings,
    ask_save_path,
    run_export,
    safe_name,
    written_note,
)
from detail.gui.busy import decode_overlay
from detail.gui.dialog import Dialog
from detail.gui.image_view import ImageView, PictureWithControls
from detail.gui.run_preview import _to_pixmap
from detail.gui.scrubber import FrameScrubber
from detail.gui.sizing import fit_to_screen, lines

#: Opening size in lines, so it follows a user who asked for larger text; capped to the screen
#: and resizable from there.
_WIDE = 70
_TALL = 46
_FOLLOWS = (
    "Follows the clip window: its Mask trail, Trail, Trail points, Align pose and Context "
    "ticks, and its clip."
)
_DECODES = (
    " This recording's record carries no frame size, so the footage is decoded to size the "
    "pane."
)


class ClipPaneWindow(Dialog):
    """``pane`` (``"masks"`` / ``"keypoints"``) of ``player``'s clip, large."""

    def __init__(self, player, pane: str, parent=None) -> None:
        """
        Raises:
            ValueError: ``pane`` is not one of ``api.clip_views.PANES``.
        """
        from detail.api.run_preview import ClipQueryPreview

        if pane not in clip_views.PANES:
            raise ValueError(f"pane must be one of {clip_views.PANES}, not {pane!r}")
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._player = player
        self._pane = pane
        # Its own decode context: the clip window's player decodes on another thread, and a
        # context's reader keeps its place in the video between calls.
        self._preview = ClipQueryPreview(player._experiment.dir)
        self._experiment = player._experiment
        self._settings = player.view_settings()
        self._clip = player.clip()
        self._bounds = player.window()
        # The panes are stored data at the record's frame size; a record with no size has
        # nothing to draw them at without the footage, so this window decodes it and says so.
        # The API refuses to guess (`clip_frame(footage=False)` raises); the window decides.
        self._footage = (
            self._clip is not None
            and self._preview.frame_size(self._clip.recording_id) is None
        )

        layout = self.body()
        self._note = QLabel(_FOLLOWS + (_DECODES if self._footage else ""))
        self._note.setObjectName("pane_follows_note")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)

        self._view = ImageView(zoomable=True)
        self._view.setObjectName(f"enlarged_{pane}")
        self._view.setMinimumSize(QSize(lines(16), lines(10)))
        self._scrubber = FrameScrubber(playable=True, loop=True)
        self._scrubber.set_decoder(self._compute)
        self._scrubber.rendered.connect(self._display)
        self._view.clicked.connect(self._toggle_play)
        decode_overlay(self._scrubber, self._view)
        self._loop = QCheckBox("Loop")
        self._loop.setObjectName("pane_loop")
        self._loop.setChecked(True)
        self._loop.setToolTip("Start again from the first frame after the last.")
        self._loop.toggled.connect(self._scrubber.set_loop)
        self._export_video = QPushButton("Export video…")
        self._export_video.setObjectName("pane_export_video")
        self._export_video.setToolTip(
            "Write this pane as an mp4: every frame on the bar, at full size."
        )
        self._export_video.clicked.connect(self.export_video)
        self._export_gif = QPushButton("Export GIF…")
        self._export_gif.setObjectName("pane_export_gif")
        self._export_gif.setToolTip(
            "Write this pane as a GIF, at the width and frame rate you choose next."
        )
        self._export_gif.clicked.connect(self.export_gif)
        # `add_control` inserts right after the step buttons, so the last added shows first.
        for control in (self._export_gif, self._export_video, self._loop):
            self._scrubber.add_control(control)
        self._block = PictureWithControls(self._view, self._scrubber)
        layout.addWidget(self._block, 1)
        self._status = QLabel("")
        self._status.setObjectName("pane_export_status")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)

        player.view_changed.connect(self._on_settings)
        player.window_changed.connect(self._on_window)
        player.clip_changed.connect(self._on_clip)
        # The clip window rebuilds its player when Advanced is toggled; this window follows a
        # player, so it closes with the one it was opened from rather than outlive it.
        player.destroyed.connect(self.close)
        self.finished.connect(self._let_go)
        self._sync(start=player.current_frame())
        self._title()
        self.resize(
            fit_to_screen(
                QSize(lines(_WIDE), lines(_TALL)), screen=parent_screen(parent)
            )
        )

    # -- following the clip window -------------------------------------------- #
    def _on_settings(self) -> None:
        self._settings = self._player.view_settings()
        self._scrubber.set_frame(self._scrubber.frame())  # redraw what is on screen

    def _on_window(self) -> None:
        self._sync(start=self.current_frame())

    def _on_clip(self) -> None:
        self._clip = self._player.clip()
        self._sync(start=self._player.current_frame())
        self._title()

    def _let_go(self, _result: int) -> None:
        """Closed: stop playing, stop following the player, and leave its register — so the
        next click opens a fresh window and nothing of this one keeps running."""
        self._scrubber.pause()
        player = self._player
        try:
            for signal, slot in (
                (player.view_changed, self._on_settings),
                (player.window_changed, self._on_window),
                (player.clip_changed, self._on_clip),
                (player.destroyed, self.close),
            ):
                try:
                    signal.disconnect(slot)
                except TypeError:
                    pass  # not connected: closed twice
            windows = getattr(player, "pane_windows", None)
        except RuntimeError:
            # The clip window closed first and Qt deleted the player (this window's parent's
            # child, like this one): every connection to it went with it, and its register.
            return
        if windows is not None and windows.get(self._pane) is self:
            windows.pop(self._pane, None)

    def _sync(self, *, start: int) -> None:
        """Span the player's window of frames and show ``start`` (clamped into it)."""
        self._bounds = self._player.window()
        first, last = self._bounds
        count = max(last - first + 1, 1)
        self._scrubber.pause()
        self._scrubber.set_total(count)
        rate = self._player.frame_rate()
        if rate > 0:
            self._scrubber.set_playback_rate(rate)
        # The player's own times for the same window: already read, so nothing is read here.
        self._scrubber.set_timestamps(self._player.window_timestamps())
        lo, hi = self._player.clip_span()
        if (hi - lo + 1) < count:
            self._scrubber.set_marks(spans=[(lo - first, hi - first)])
        else:
            self._scrubber.clear_marks()
        self._scrubber.set_frame(min(max(int(start) - first, 0), count - 1))

    def _title(self) -> None:
        label = self._clip.label if self._clip is not None else ""
        self.setWindowTitle(clip_views.pane_title(self._pane, label))

    # -- playing --------------------------------------------------------------- #
    def _compute(self, step: int):
        clip, settings, first = self._clip, self._settings, self._bounds[0]
        footage = self._footage
        if clip is None:
            return None
        frame = first + int(step)
        return frame_source.decode(  # off-thread (scrubber decoder)
            lambda: self._preview.clip_views(
                clip.recording_id,
                clip.embed_run_id,
                clip.track_id,
                frame,
                settings,
                footage=footage,
            )
        )

    def _display(self, _step: int, view) -> None:
        if view is None:
            return
        views, reason = view
        if reason is not None:
            self._view.clear_frame(frame_source.unavailable_text(reason))
            return
        picture = clip_views.pane_picture(views, self._pane)
        if picture is None:
            self._view.clear_frame("No keypoints for this run")
            self._view.setToolTip(views.pose_note or "")
            return
        self._view.set_frame(_to_pixmap(picture))
        self._block.set_aspect(self._view.pixmap_size())

    def _toggle_play(self) -> None:
        button = self._scrubber.play_button()
        if button is not None:
            button.toggle()

    def current_frame(self) -> int:
        """The global frame under this window's playhead."""
        return int(self._bounds[0] + self._scrubber.frame())

    def scrubber(self) -> FrameScrubber:
        return self._scrubber

    def view(self) -> ImageView:
        return self._view

    def settings(self) -> clip_views.ClipViewSettings:
        """The view ticks this window is drawing with (the clip window's, as last changed)."""
        return self._settings

    def frames(self) -> tuple[int, int]:
        """The inclusive global frames on this window's bar — what an export writes."""
        return self._bounds

    def decodes_footage(self) -> bool:
        """Whether this window has to decode the footage to size its pane (no recorded
        frame size); the note says so when it does."""
        return self._footage

    # -- exporting ------------------------------------------------------------- #
    def frames_note(self) -> str:
        """Which frames an export holds, in words."""
        context = self._player.context_s()
        first, last = self._bounds
        count = last - first + 1
        if context:
            return (
                f"The clip with {context:g} s of context either side (Context is on in the "
                f"clip window): {count:,} frames."
            )
        return (
            f"The clip itself, {count:,} frames. Turn Context on in the clip window to "
            "include footage either side."
        )

    def export_video(self) -> None:
        path = ask_save_path(self, "Export video", self._suggested(MP4), MP4)
        if path is not None:
            self.write(path)

    def export_gif(self) -> None:
        answer = ask_gif_settings(self, self.frames_note())
        if answer is None:
            return
        path = ask_save_path(self, "Export GIF", self._suggested(GIF), GIF)
        if path is not None:
            width, fps = answer
            self.write(path, max_width=width, fps=fps)

    def write(self, path, *, max_width: int | None = None, fps: float | None = None):
        """Export this pane to ``path`` (``.mp4`` / ``.gif``) off the UI thread, with the
        settings and frames on screen now."""
        clip, settings, (first, last) = self._clip, self._settings, self._bounds
        experiment_dir, pane = self._experiment.dir, self._pane

        def work(progress):
            return clip_views.export_clip_pane(  # off-thread (run_progress)
                experiment_dir,
                recording_id=clip.recording_id,
                embed_run_id=clip.embed_run_id,
                track_id=clip.track_id,
                first=first,
                last=last,
                pane=pane,
                settings=settings,
                path=path,
                max_width=max_width,
                fps=fps,
                progress=progress,
            )

        return run_export(
            self,
            work,
            on_written=lambda written: self._status.setText(written_note(written)),
            label=f"Exporting the {pane} pane…",
        )

    def _suggested(self, kind: str) -> str:
        label = self._clip.label if self._clip is not None else "clip"
        return f"{safe_name(label)}_{self._pane}.{kind}"


def parent_screen(widget):
    """The screen a window will open on: its parent's, or ``None`` with no parent yet."""
    if widget is None:
        return None
    try:
        return widget.screen()
    except (AttributeError, RuntimeError):
        return None


def open_pane_window(player, pane: str, parent) -> ClipPaneWindow:
    """Open ``pane`` of ``player``'s clip large, **non-modally**: it is watched beside the clip
    window, whose ticks it follows, so that window must stay usable (``requirements.py``'s
    ``_NON_MODAL_DIALOGS`` says the same). One window per pane: while one is open, a second
    click raises it rather than opening another."""
    windows = getattr(player, "pane_windows", None)
    if windows is None:
        windows = player.pane_windows = {}
    existing = windows.get(pane)
    if existing is not None:
        try:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return existing
        except (
            RuntimeError
        ):  # the C++ side is gone: a closed window not yet unregistered
            windows.pop(pane, None)
    window = ClipPaneWindow(player, pane, parent=parent)
    windows[pane] = window
    window.show()
    return window
