"""The clip window's Advanced player (§22): the footage with the masks and the keypoints beside
it, the strips under the slider, and the view ticks that say what to draw.

A subclass of the plain player over its five hooks (``behavior_query._ClipPlayer``), not a
``panels=`` flag on it: eleven surfaces embed the plain player and none of them wants a trail
picker, three pictures or strips, so what is only this window's state lives only here. The
plain player's tests are the regression net; delete the overrides and this class is the plain
player again.

Every decode goes through one call, ``ClipQueryPreview.clip_views`` with a
``ClipViewSettings`` snapshot taken on the UI thread; the panes read the result, the ticks are
greyed from what the frame says it has (a run with no keypoints greys **Keypoints**, **Align
pose** and **Trail points**, each with the reason). The strips are ``ClipQueryPreview.clip_curves``
over the window on the bar, read off the UI thread; a later window supersedes a read in flight.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QSizePolicy, QWidget

from detail.api.clip_views import ClipViewSettings
from detail.gui.behavior_query import ClipSpec, _ClipPlayer
from detail.gui.run_preview import _to_pixmap

#: The clip popup's Keypoints toggle, while there are keypoints to draw.
POINTS_TIP = "Draw the clip's animal's keypoints on the footage: the keypoint run the embedding used."
ALIGN_TIP = (
    "Turn the Keypoints pane so the clip's animal always faces up, centred and at its usual "
    "size, so postures compare from clip to clip whichever way the animal faced. With Trail "
    "on, the pane's trails are then each point's path relative to the body."
)
TRAIL_TIP = (
    "Draw where the clip's animal has just been, over the last half second: each keypoint's "
    "path in its own colour, on the footage and on the Keypoints pane (choose which with "
    "Trail points). A run with no keypoints draws the centre of the animal instead."
)
MASK_TRAIL_TIP = (
    "Draw the clip's animal's masks from the last half second under the current one, fading "
    "with age, to see how its outline moved and changed."
)
LOOP_TIP = "Play the clip itself over and over, without the context either side."


class AdvancedClipPlayer(_ClipPlayer):
    """The plain player, plus the panes, the strips and the view ticks."""

    #: A view tick changed (Masks, Keypoints, Trail, Trail points, Mask trail, Align pose):
    #: what `view_settings` returns is different now. An enlarged pane follows it.
    view_changed = pyqtSignal()
    #: `set_clip` showed another clip.
    clip_changed = pyqtSignal()

    # -- the hooks ------------------------------------------------------------- #
    def _add_view_controls(self, _row) -> None:
        """The view ticks, as **one unit of their own** in the scrubber's flowing row, after
        Context and Masks rather than inside their unit.

        Inside it, the row's one-line width was ~1340px at the system font, and a window
        narrower than that (a 1440px laptop screen) wrapped the only thing left over — the frame
        counter — onto a line by itself. As a unit, the ticks are what moves to the second line,
        together, with the counter at its right end; on a row that fits, the picture is the
        same.
        """
        from detail.gui.skeleton_points import SkeletonPointsButton

        #: The view ticks' unit in the controls row.
        self.view_ticks = QWidget()
        self.view_ticks.setObjectName("clip_view_ticks")
        row = QHBoxLayout(self.view_ticks)
        row.setContentsMargins(0, 0, 0, 0)

        def tick(text, name, tip):
            box = QCheckBox(text)
            box.setObjectName(name)
            box.setToolTip(tip)
            box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            box.toggled.connect(self._redraw)
            row.addWidget(box)
            return box

        self._points = tick("Keypoints", "clip_points", POINTS_TIP)
        self._trail = tick("Trail", "clip_trail", TRAIL_TIP)
        # Which points trail: the picker in a menu, so the row stays one line. Its skeleton
        # arrives with the first decoded frame.
        self._trail_button = SkeletonPointsButton("Trail points")
        self._trail_button.setObjectName("clip_trail_points")
        self._trail_button.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self._trail_button.changed.connect(self._on_trail_points)
        self._trail_names: list[str] | None = None  # every point
        self._trail_skeleton: tuple | None = None
        row.addWidget(self._trail_button)
        self._mask_trail = tick("Mask trail", "clip_mask_trail", MASK_TRAIL_TIP)
        self._align = tick("Align pose", "clip_align", ALIGN_TIP)
        self._loop_clip = QCheckBox("Loop clip")
        self._loop_clip.setObjectName("clip_loop")
        self._loop_clip.setToolTip(LOOP_TIP)
        self._loop_clip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._loop_clip.toggled.connect(self._on_loop_clip)
        row.addWidget(self._loop_clip)
        # Called before the base adds Context / Masks and Export after Next, each inserted
        # straight after Next — so this lands after both.
        self._scrubber.add_control(self.view_ticks)

    def _picture_widget(self) -> QWidget:
        from detail.gui.clip_panes import ClipPanes
        from detail.gui.clip_strips import ClipStrips

        self._aligned = False
        self.pane_windows: dict[str, QWidget] = {}
        self._curves_generation = 0
        self._panes = ClipPanes(self._src)
        self._panes.enlarge_wanted.connect(self.open_pane)
        self._strips = ClipStrips(self._scrubber)
        self._scrubber.set_underlay(self._strips)
        self._scrubber.frame_changed.connect(self._strips.set_frame)
        self.window_changed.connect(self._load_curves)
        return self._panes

    def _decode(self, clip: ClipSpec, frame: int):
        # One decode for all three pictures: the masks pane needs the masks whatever the
        # toggles say, so there is no cheaper plain path here. The settings are read on the
        # UI thread and handed to the decoder as a value.
        settings = self.view_settings()
        return self._preview.clip_views(  # off-thread (scrubber decoder)
            clip.recording_id, clip.embed_run_id, clip.track_id, frame, settings
        )

    def _show_payload(self, views) -> None:
        self._src.set_frame(_to_pixmap(views.frame))
        self._panes.show_views(views)
        # Read off the frame, not asked of the run on the UI thread: a run with no keypoints
        # greys the toggle and says why; so does alignment, which also needs a heading.
        has_points = views.pose is not None
        self._points.setEnabled(has_points)
        self._points.setToolTip(POINTS_TIP if has_points else (views.pose_note or ""))
        self._align.setEnabled(views.align_note is None)
        self._align.setToolTip(views.align_note or ALIGN_TIP)
        self._show_trail_controls(views)
        self._panes.set_frame_aspect(self._src.pixmap_size())
        self._block.set_aspect(self._panes.aspect_size(self._src.pixmap_size()))

    def _show_unavailable(self, text: str) -> None:
        super()._show_unavailable(text)
        self._panes.clear("")

    def _clip_only(self) -> bool:
        return self._walk_edges or self._loop_clip.isChecked()

    def _redraw(self, *_args) -> None:
        super()._redraw()
        self.view_changed.emit()

    # -- what the window shows ------------------------------------------------- #
    def set_clip(self, clip: ClipSpec, *, autoplay: bool | None = None) -> None:
        super().set_clip(clip, autoplay=autoplay)
        self._load_curves()
        self.clip_changed.emit()

    def clip(self) -> ClipSpec | None:
        """The clip on screen."""
        return self._clip

    def frame_rate(self) -> float:
        """The clip's frames per second, as the player plays it."""
        return self._frame_rate(self._clip) if self._clip is not None else 0.0

    def window_timestamps(self):
        """The recording's ``t_ns`` over the window on the bar, or ``None`` — for a companion
        view spanning the same frames (an enlarged pane)."""
        return self._window_timestamps()

    def view_settings(self) -> ClipViewSettings:
        """The view ticks as plain values — what every picture is drawn with, read on the UI
        thread and handed to a decoder."""
        names = self._trail_names
        return ClipViewSettings(
            masks_on_frame=self._masks.isChecked(),
            points_on_frame=self._points.isChecked(),
            trail=self._trail.isChecked(),
            trail_points=None if names is None else tuple(names),
            mask_trail=self._mask_trail.isChecked(),
            pose_aligned=self._align.isChecked(),
            aligned=self._aligned,
        )

    def set_aligned(self, on: bool) -> None:
        """Draw every picture **Aligned** (the clip window's orientation switch), or as
        recorded."""
        if bool(on) != self._aligned:
            self._aligned = bool(on)
            self._redraw()

    def is_aligned(self) -> bool:
        return self._aligned

    def strips(self):
        """The strips under the slider."""
        return self._strips

    def panes(self):
        """The three-picture widget."""
        return self._panes

    def trail_points(self) -> list[str] | None:
        """The keypoints that trail, by name (``None``: every point)."""
        return self._trail_names

    def open_pane(self, pane: str):
        """Open ``pane`` (``"masks"`` / ``"keypoints"``) large in a window of its own, which
        follows this player's clip and ticks (``gui/clip_pane_window.py``); the same window
        again while one is open. Returns it."""
        from detail.gui.clip_pane_window import open_pane_window

        # `QWidget.window`, not `self.window()`: this class's `window()` is the frames it spans.
        return open_pane_window(self, pane, QWidget.window(self))

    # -- the ticks ------------------------------------------------------------- #
    def _on_loop_clip(self, on: bool) -> None:
        self._scrubber.set_loop(on)
        self._apply_playback_range()
        if on:
            self._scrubber.set_frame(max(self._clip_span[0] - self._bounds[0], 0))

    def _on_trail_points(self) -> None:
        """Hold the picker's choice as a plain value for the decode, then redraw."""
        self._trail_names = self._trail_button.value()
        self._redraw()

    def _show_trail_controls(self, views) -> None:
        """Give the picker the clip's skeleton, or grey the button with the reason when there
        is none; and say on the Trail tick what it is drawing for this clip."""
        skeleton = views.skeleton
        if skeleton is None:
            self._trail_button.set_reason(
                (views.pose_note or "No keypoints for this run.")
                + " Trail follows the centre of the animal instead."
            )
        else:
            self._trail_button.set_reason(None)
            shape = tuple(skeleton.names)
            if shape != self._trail_skeleton:
                self._trail_skeleton = shape
                self._trail_button.set_skeleton(shape)
                self._trail_names = self._trail_button.value()
        self._trail.setToolTip(views.trail_note or TRAIL_TIP)

    # -- the strips ------------------------------------------------------------ #
    def _load_curves(self) -> None:
        """Read the strips for the window on screen, off the UI thread; a later window
        supersedes an earlier read still in flight."""
        if self._clip is None:
            return
        from detail.gui.workers import BackgroundCall

        self._curves_generation += 1
        generation = self._curves_generation
        clip, (first, last) = self._clip, self._bounds
        lo, hi = self._clip_span[0] - first, self._clip_span[1] - first
        self._strips.set_clip_span((lo, hi))
        call = BackgroundCall(
            lambda: self._preview.clip_curves(  # off-thread (BackgroundCall)
                clip.recording_id, clip.embed_run_id, clip.track_id, first, last
            ),
            receiver=self,
        )
        call.done.connect(lambda curves: self._show_curves(generation, curves))
        call.failed.connect(
            lambda exc: generation == self._curves_generation
            and self._strips.set_message(f"No strips: {exc}")
        )
        call.start()

    def _show_curves(self, generation: int, curves) -> None:
        if generation != self._curves_generation:
            return  # a newer window was asked for
        self._strips.set_curves(curves)
        self._strips.set_frame(self._scrubber.frame())


class PlainClipPlayer(_ClipPlayer):
    """The plain player, with the clip window's orientation switch: **Aligned** turns the
    footage (and the mask on it) so the body points up, through the same read and turn as the
    Advanced player (``ClipQueryPreview.clip_views``); as recorded, it is the plain player's
    own fast path, untouched."""

    def _picture_widget(self) -> QWidget:
        self._aligned = False
        return super()._picture_widget()

    def set_aligned(self, on: bool) -> None:
        if bool(on) != self._aligned:
            self._aligned = bool(on)
            self._redraw()

    def is_aligned(self) -> bool:
        return self._aligned

    def view_settings(self) -> ClipViewSettings:
        """What the picture is drawn with: the Masks tick and the orientation."""
        return ClipViewSettings(
            masks_on_frame=self.masks_shown(),
            points_on_frame=False,
            aligned=self._aligned,
        )

    def _decode(self, clip: ClipSpec, frame: int):
        if not self._aligned:
            return super()._decode(clip, frame)
        settings = self.view_settings()
        views = self._preview.clip_views(  # off-thread (scrubber decoder)
            clip.recording_id, clip.embed_run_id, clip.track_id, frame, settings
        )
        if views is None:
            raise ValueError("this frame could not be read")
        return (views.frame, views.actual, views.t_ns)
