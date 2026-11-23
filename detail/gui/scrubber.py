"""A reusable frame scrubber: slider + ◀/▶ + an editable frame-number readout,
with keyboard stepping that **accelerates while an arrow key is held** and a
**background, latest-only decode** so scrubbing stays responsive on long video.

The mouse-dragged slider is ~1 frame/pixel only on short clips; on a 400k-frame
video it is far too coarse to land near a target. Keyboard stepping with
hold-to-accelerate gives precision (a tap moves one frame) and a hold that follows the animal
up to ``HOLD_MAX_SPEED`` (3x real time); Shift+arrows, PageUp/PageDown and typing a number in
the readout are for distance.

Decode runs **off the UI thread**, rendering only the *latest* requested frame
(intermediate ones are dropped). A deliberate spam of clicks therefore costs one
decode of the final frame, not N synchronous decodes — and a slow decode (e.g. a
multi-GB source video) never freezes the GUI.

Hosts embed a ``FrameScrubber``, call ``set_total`` / ``set_frame``, register a
``set_decoder(compute_fn)`` (``compute_fn(frame) -> payload`` runs on a worker
thread — **no Qt** in it), and connect:
- ``frame_changed(int)`` — emitted immediately on every change; for cheap UI
  (the host's current-frame state, a strip/timeline cursor).
- ``rendered(int, object)`` — emitted on the UI thread with the worker's payload;
  for the expensive display (build the ``QPixmap`` here, on the UI thread).
Extra controls (e.g. a "Masks" checkbox) go in the control row via ``add_control``.

Two more things a host hands over rather than the widget working out: ``set_timestamps`` (the
recording's ``t_ns`` per scrub position, so the row can say what time the frame is) and
``set_marks`` (ticks and spans drawn on the slider — a pending start, the clip inside its context).
"""

import time
from collections import deque
from collections.abc import Callable, Sequence

import numpy as np

from detail.gui import plot_theme
from detail.gui.flow import FlowLayout
from detail.gui.qt import (
    QColor,
    QComboBox,
    QEvent,
    QHBoxLayout,
    QIntValidator,
    QLabel,
    QLineEdit,
    QObject,
    QPainter,
    QPalette,
    QPen,
    QPointF,
    QPolygonF,
    QPushButton,
    QRectF,
    QRunnable,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QThreadPool,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)
from detail.gui.sizing import text_width
from detail.gui.timefmt import TIME_WIDEST, format_time

_STEP_CAP = 128  # max single-press step at full acceleration

#: The fastest a held step button or arrow key moves, as a multiple of the video's own rate. The
#: doubling step alone reached 71x on the buttons and ~140x on the keys within two seconds, which
#: is skipping, not looking — a hold is for following an animal, and the slider, Shift+arrows and
#: the frame number are there for distance. Three is the cap the boundary editor's edge walk
#: already uses (`behavior_query._EDGE_MAX_STEP`), for the same reason.
HOLD_MAX_SPEED = 3.0

#: The forward half of the speed menu, as multiples of the surface's own rate. YouTube's set,
#: because a menu people already know beats one we invent.
_FORWARD_SPEEDS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0)

#: The speed menu. A negative speed plays backward: reverse is a *speed*, as VLC models it
#: (its Reverse control negates the rate), rather than a second Play button — one control starts
#: and stops playback, and the menu says which way and how fast.
PLAYBACK_SPEEDS = tuple(-s for s in reversed(_FORWARD_SPEEDS)) + _FORWARD_SPEEDS

#: Fallback rate for a playable surface whose host never called :meth:`set_playback_rate`. Only a
#: floor against a zero interval — every playable site today has a real rate to give.
DEFAULT_PLAYBACK_FPS = 30.0

#: How far below the requested rate the achieved one has to fall before it is shown. Decode is
#: latest-only, so a slow source drops frames rather than falling behind: the playhead keeps real
#: time and the picture updates less often. Saying "1x" through that is the lie this catches.
_RATE_HONESTY = 0.85

#: Rendered-frame timestamps kept for the achieved-rate estimate. Enough to be steady, short enough
#: to react within about a second at ordinary rates.
_RATE_WINDOW = 15

#: The widest thing the achieved-rate note can say, which is what its reserved width is sized
#: from. Three digits covers any rate a decode reaches; a fourth would mean the note is moot.
_RATE_NOTE_WIDEST = "999 fps"


def accelerated_step(held_repeats: int) -> int:
    """Per-press frame step given how many auto-repeats deep the held key is.

    ``0`` (the initial press) → 1; the step doubles every 4 repeats so a tap is
    frame-precise and a sustained hold covers large distances, capped so it never
    overshoots wildly: 1,1,1,1,2,2,2,2,4,…,128.
    """
    return min(1 << (max(0, held_repeats) // 4), _STEP_CAP)


def held_step(held_repeats: int, fps: float, seconds_since_last: float | None) -> int:
    """``accelerated_step``, capped so a hold never runs faster than ``HOLD_MAX_SPEED``.

    The cap is a **rate**, not a step count: frames per step = cap × fps × the time since the
    previous step. So it means the same 3x on 25, 30 or 60 fps video, and on a keyboard whose
    auto-repeat is 30 ms as on a button that repeats every 60. ``None`` (the first press) is
    one frame.
    """
    step = accelerated_step(held_repeats)
    if seconds_since_last is None:
        return 1
    # Clamped above: a pause between two presses is not a long, fast step.
    elapsed = min(max(seconds_since_last, 0.0), 0.25)
    return max(1, min(step, int(HOLD_MAX_SPEED * fps * elapsed)))


class _DecodeFailed:
    """Sentinel payload for an unexpected exception in a decode task — the frame
    is skipped (the prior image stays) rather than crashing the worker/GUI."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class _DecodeSignals(QObject):
    done = Signal(int, object)  # (frame, payload) — delivered on the UI thread


class _DecodeTask(QRunnable):
    """Run ``compute_fn(frame)`` on a thread-pool worker; emit the payload back."""

    def __init__(self, frame: int, compute_fn, signals: _DecodeSignals) -> None:
        super().__init__()
        self._frame = frame
        self._compute_fn = compute_fn
        self._signals = signals

    def run(self) -> None:
        try:
            payload = self._compute_fn(self._frame)
        except Exception as exc:  # noqa: BLE001 — boundary: never crash the pool
            payload = _DecodeFailed(exc)
        try:
            self._signals.done.emit(self._frame, payload)
        except RuntimeError:
            pass  # the scrubber (with its signals) was torn down mid-decode — drop it


class _FrameEntry(QLineEdit):
    """Frame-number field that **consumes** Return/Enter (after emitting
    ``returnPressed``) so pressing Enter to jump doesn't also fire a parent
    dialog's default button — which would close the dialog."""

    def keyPressEvent(self, event) -> None:
        super().keyPressEvent(event)  # emits returnPressed → the jump
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()  # stop it bubbling to the dialog's default button


#: Half-width of a tick's arrowhead, and how far a span bracket's ends reach inward, in pixels.
_TICK_HEAD_PX = 5
_BRACKET_CAP_PX = 4
#: Height of a band (drawn under the groove), in pixels.
_SPAN_PX = 4

#: How much of the window colour is laid over the groove outside a span. Enough that the context
#: reads as *not the clip* at a glance, little enough that the played fill still shows through.
_CONTEXT_DIM_ALPHA = 170


class _MarkedSlider(QSlider):
    """A slider that draws ticks and spans on its own groove.

    On the slider rather than in a strip beneath it: a strip that appears when the first mark does
    moves every control under it, and one reserved on every surface is empty space on the ten
    that never mark anything. Painted after the style, so the marks sit over the groove and under
    nothing a person has to grab: a span is a pair of brackets with the groove outside them dimmed —
    how QuickTime and iMovie show a trim selection — and a tick is a thin line with its arrowhead
    above the groove, so the handle stays whole wherever it sits.
    """

    def __init__(self, orientation) -> None:
        super().__init__(orientation)
        self._ticks: tuple[int, ...] = ()
        self._spans: tuple[tuple[int, int], ...] = ()
        self._bands: tuple[tuple[int, int, str | None, bool], ...] = ()

    def set_marks(self, ticks, spans, bands=()) -> None:
        self._ticks = tuple(int(t) for t in ticks)
        self._spans = tuple((int(a), int(b)) for a, b in spans)
        self._bands = tuple(
            (int(first), int(last), color, bool(hollow))
            for first, last, color, hollow in bands
        )
        self.update()

    def bands(self) -> tuple[tuple[int, int, str | None, bool], ...]:
        return self._bands

    def marks(self) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        return self._ticks, self._spans

    def travel(self) -> tuple[int, int]:
        """``(left, width)`` in pixels of the handle's centre travel — see ``handle_travel``."""
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()
        groove = style.subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        handle = style.subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        # Qt centres an even-width handle a half-pixel left of its rect's midpoint, so the
        # inset is (w-1)//2 — using w//2 puts the strip a pixel right of the handle at
        # every value.
        half = (handle.width() - 1) // 2
        return groove.x() + half, max(groove.width() - handle.width(), 1)

    def x_of(self, value: int) -> float:
        """The pixel column the handle's centre sits on at ``value``."""
        left, width = self.travel()
        span = max(self.maximum() - self.minimum(), 1)
        clamped = min(max(value, self.minimum()), self.maximum())
        return left + width * (clamped - self.minimum()) / span

    def _groove(self) -> QRectF:
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        return QRectF(
            self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider,
                opt,
                QStyle.SubControl.SC_SliderGroove,
                self,
            )
        )

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().paintEvent(event)
        if not self._ticks and not self._spans and not self._bands:
            return
        theme = plot_theme.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        if self._bands:
            self._paint_bands(painter, theme)
        if self._spans:
            self._paint_spans(painter, theme)
        painter.setBrush(QColor(theme.hue("vermillion")))
        for tick in self._ticks:
            x = self.x_of(tick)
            head = _TICK_HEAD_PX
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x - head, 0.0),
                        QPointF(x + head, 0.0),
                        QPointF(x, head * 1.4),
                    ]
                )
            )
            painter.drawRect(QRectF(x - 1, 0.0, 2.0, float(self.height())))
        painter.end()

    def _paint_bands(self, painter: QPainter, theme) -> None:
        """A coloured band under the groove per marked range — filled, or outlined when hollow.

        Not a span: a span is *the* clip inside its context, so everything outside it is dimmed.
        Bands are many ranges over a whole recording (the example dialog's clips, one colour per
        behavior), and dimming outside them would grey out nearly all of it.
        """
        groove = self._groove()
        band_top = min(groove.bottom() + 1.0, self.height() - _SPAN_PX)
        for first, last, color, hollow in self._bands:
            fill = QColor(color or theme.hue("green"))
            x0, x1 = self.x_of(first), self.x_of(last)
            rect = QRectF(x0, band_top, max(x1 - x0, 2.0), float(_SPAN_PX))
            if hollow:
                painter.setPen(QPen(fill, 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect.adjusted(0.75, 0.75, -0.75, -0.75))
                painter.setPen(Qt.PenStyle.NoPen)
            else:
                painter.setBrush(fill)
                painter.drawRect(rect)

    def _paint_spans(self, painter: QPainter, theme) -> None:
        """Dim the groove outside the spans, then bracket each one.

        A 4px green line under the groove was the first version, and it was reported as "a solid
        colour bar I don't understand": nothing about a line says *this part is the clip*. Dimming
        what is not the clip and bracketing what is, is the trim picture people already know.
        """
        groove = self._groove().adjusted(0, -2, 0, 2)
        dim = QColor(self.palette().color(QPalette.ColorRole.Window))
        dim.setAlpha(_CONTEXT_DIM_ALPHA)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(dim)
        edges = sorted((self.x_of(a), self.x_of(b)) for a, b in self._spans)
        cursor = groove.left()
        for x0, x1 in edges:
            if x0 > cursor:
                painter.drawRect(
                    QRectF(cursor, groove.top(), x0 - cursor, groove.height())
                )
            cursor = max(cursor, x1)
        if groove.right() > cursor:
            painter.drawRect(
                QRectF(cursor, groove.top(), groove.right() - cursor, groove.height())
            )
        pen = QPen(QColor(theme.hue("orange")), 2)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        top, bottom = 1.0, float(self.height() - 1)
        cap = _BRACKET_CAP_PX
        for x0, x1 in edges:
            painter.drawPolyline(
                QPolygonF(
                    [
                        QPointF(x0 + cap, top),
                        QPointF(x0, top),
                        QPointF(x0, bottom),
                        QPointF(x0 + cap, bottom),
                    ]
                )
            )
            painter.drawPolyline(
                QPolygonF(
                    [
                        QPointF(x1 - cap, top),
                        QPointF(x1, top),
                        QPointF(x1, bottom),
                        QPointF(x1 - cap, bottom),
                    ]
                )
            )


class FrameScrubber(QWidget):
    """Slider + step buttons + editable readout; hold-to-accelerate keys; background decode.

    ``loop=True`` replays instead of stopping at the end — right for a clip somebody is judging,
    wrong for a recording they are watching through.

    ``playable=True`` adds play/pause and a speed menu. It is **required** and has no default:
    whether a surface can be *played* is a fact about that surface rather than a preference, and
    a default spelling "as before" is where a missing fact hides — defaulting False is what left
    nine surfaces with no player for months.

    Refusing it takes a reason, and the two that do have one. Quick segment's sampled view plays
    only where its positions form a sequence (``_uniform_stride``); a stream preview with no video
    has no frames behind the slider at all. Both compute the flag rather than hardcoding it.

    (An earlier version of this note claimed the three ``arena`` dialogs must refuse playback
    because ``_on_scrub`` writes to the current keyframe. That was wrong — ``_capture_current``
    only writes to a frame that is *already* a keyframe, so scrubbing ordinary frames writes
    nothing. They play, starting paused.)

    Playback keeps **real time, not every frame**. The decode is latest-only, so a source the
    machine cannot keep up with drops frames while the playhead stays on the clock; the control row
    then says the rate actually achieved rather than claiming the nominal one.
    """

    frame_changed = Signal(int)  # immediate, every change — cheap UI updates
    rendered = Signal(int, object)  # UI thread; the worker payload to display
    scrubbed = (
        Signal()
    )  # a *manual* frame move (button/slider/key/typed) — not autoplay
    busy_changed = Signal(bool)  # a decode is in flight (True) / caught up (False);
    # hosts drive a "Decoding frame…" overlay off it, so first-load and slow scrubs
    # aren't a blank pane. Stays True across a scrub burst (re-dispatch), one False at rest.

    def __init__(
        self,
        parent=None,
        label_formatter=None,
        allow_frame_entry=True,
        *,
        playable: bool,
        loop: bool = False,
    ):
        super().__init__(parent)
        self._total = 0
        self._frame = 0
        self._held = 0  # auto-repeat depth of the currently held arrow key
        self._label_formatter = label_formatter or (lambda v, last: f"{v} / {last}")

        # `playable` has **no default**, on purpose. It decides whether the output is correct
        # rather than how it looks, and CLAUDE.md records what a default costs there: "a default
        # spelling 'as before' is where a missing fact hides". Defaulting True puts a play button
        # on the calibration dialog; defaulting False is the opt-in that left nine surfaces with
        # no player for months. Twelve call sites, one word each, and the arena dialogs' refusal
        # becomes a stated decision instead of an omission.
        self._playable = bool(playable)
        # A genuine choice, unlike `playable`: a clip a person is judging is watched over and
        # over, and a recording is watched through. Neither is more correct, so this keeps a
        # default and is listed in the widget's allowlist with that reason.
        self._loop = bool(loop)
        # The frames playback runs between, when it is not the whole span. A clip player shows a
        # *window* around its clip so an edge can be dragged outward, and playing that window
        # start to finish would play footage either side of the thing being judged.
        self._play_first: int | None = None
        self._play_last: int | None = None
        self._fps = DEFAULT_PLAYBACK_FPS
        self._speed = 1.0
        self._play_btn: QPushButton | None = None
        # When the previous held step landed, for `held_step`'s rate cap.
        self._last_step_at: float | None = None
        self._timestamps: np.ndarray | None = None
        self._speed_box: QComboBox | None = None
        self._rate_note: QLabel | None = None
        self._play_timer: QTimer | None = None
        # Playback is *armed* until a frame actually arrives. Starting the clock against a pane
        # that has not decoded yet counts frames nobody saw, and on an undecodable source it
        # would run to the end showing nothing.
        self._start_on_first_frame = False
        self._has_rendered = False
        self._render_times: deque[float] = deque(maxlen=_RATE_WINDOW)

        # background decode (latest-frame-only)
        self._compute_fn: Callable[[int], object] | None = None
        self._pending: int | None = None
        self._busy = False
        # Unparented on purpose: a task holds a ref to these signals, so they
        # outlive the scrubber if the host (dialog) closes mid-decode — the task
        # can still emit safely, and Qt drops the now-dead receiver's slot.
        self._signals = _DecodeSignals()
        self._signals.done.connect(self._on_decoded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._slider = _MarkedSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.installEventFilter(self)  # intercept arrows before the slider
        # Elapsed left of the slider, total right — where VLC, mpv and every player people
        # already know put them. Hidden, not blank and not guessed, on a surface whose host gave
        # no timestamp map: frame ÷ fps drifts on any file whose header rate is not its real one,
        # and a confident wrong time is worse than none. Each holds its widest width, so the
        # slider does not change length as the minutes gain a digit.
        self._elapsed = QLabel("")
        self._elapsed.setObjectName("scrubber_elapsed")
        self._elapsed.setToolTip(
            "Time of the frame on screen, from the start of the video"
        )
        self._total_time = QLabel("")
        self._total_time.setObjectName("scrubber_total_time")
        # The time at the bar's right end — the video's length on a whole-recording view, the end
        # of the window on a clip player, which is why it does not claim to be either.
        self._total_time.setToolTip("Time at the end of the bar")
        for label, align in (
            (self._elapsed, Qt.AlignmentFlag.AlignRight),
            (self._total_time, Qt.AlignmentFlag.AlignLeft),
        ):
            label.setFixedWidth(text_width(TIME_WIDEST))
            label.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            label.setVisible(False)
        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(0, 0, 0, 0)
        slider_row.addWidget(self._elapsed)
        slider_row.addWidget(self._slider, 1)
        slider_row.addWidget(self._total_time)
        layout.addLayout(slider_row)

        self._btn_held = (
            0  # auto-repeat depth of a held ◀/▶ button (mirrors keyboard _held)
        )
        # Reset the hold depth once the button stops repeating. `pressed`/`released` can't mark
        # the end of a hold — Qt auto-repeat fires both on every tick, so a `pressed` reset
        # would zero the depth each tick and it would never accelerate. This idle timer, longer
        # than the repeat interval and restarted on every step, only fires after the ticks stop.
        self._btn_idle = QTimer(self)
        self._btn_idle.setSingleShot(True)
        self._btn_idle.setInterval(150)
        self._btn_idle.timeout.connect(self._reset_button_hold)
        # A flow, not a box. The row is handed the video's width by `PictureWithControls`,
        # which positions its children with `setGeometry` and so enforces no minimum — measured,
        # the row needs 932px at the smallest interface font and 1144px at 12pt, and below 878px
        # Qt elides the labels ("Prev frame" -> "ev fram"). Wrapping keeps every control and
        # every label whole and costs a second line only while the pane is too narrow.
        # `gui/flow.py` has the rest, including why the trailing `addStretch` still works.
        self._controls = FlowLayout(spacing=6)
        if self._playable:
            self._build_playback_controls()
        # Named rather than arrowed. A "▶ Play" button beside a bare "▶" is two meanings on one
        # glyph, and it read as ambiguous the first time anyone opened the app — while every GUI
        # test passed, because both are labelled and both are wired. The house style is text
        # labels anyway ("Icons: none. Text labels only").
        prev = QPushButton("Prev frame")
        nxt = QPushButton("Next frame")
        # An object name, not just a label. Allowlists that walk a tree key on the name where
        # there is one, so a control that keeps its identity across a wording change keeps its
        # entry — renaming these two from "◀"/"▶" broke two snapshot guards that had keyed on
        # the glyph, which is the recorded class "key an allowlist by identity, not by label".
        prev.setObjectName("scrubber_prev_frame")
        nxt.setObjectName("scrubber_next_frame")
        self._prev = prev
        self._nxt = nxt  # `add_control` inserts after this, whatever precedes it
        for b, direction in ((prev, -1), (nxt, 1)):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keep arrow keys on the scrubber
            b.setAutoRepeat(True)  # holding repeats + accelerates, like the arrow keys
            b.setAutoRepeatDelay(300)
            b.setAutoRepeatInterval(60)
            b.clicked.connect(lambda _checked=False, d=direction: self._button_step(d))
            self._controls.addWidget(b)
        self._controls.addStretch()
        # One unit, so the flowing row never wraps between a frame number and its total.
        readout = QWidget()
        readout_row = QHBoxLayout(readout)
        readout_row.setContentsMargins(0, 0, 0, 0)
        readout_row.setSpacing(4)
        # readout: an editable "frame # / N" for frame scrubbers, else a static label
        if allow_frame_entry:
            self._entry = _FrameEntry("0")
            self._entry.setValidator(QIntValidator(0, 0, self._entry))
            self._entry.setMaximumWidth(72)
            self._entry.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._entry.returnPressed.connect(self._on_entry)
            self._max_label = QLabel("/ 0")
            readout_row.addWidget(self._entry)
            readout_row.addWidget(self._max_label)
            self._label = None
        else:
            self._entry = None
            self._label = QLabel("0 / 0")
            readout_row.addWidget(self._label)
        self._controls.addWidget(readout)
        self._readout = readout
        layout.addLayout(self._controls)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # The row's height is now a function of its width, so a host that reserves space for
        # this widget has to ask rather than read `sizeHint().height()`. `QWidget` does not
        # derive either of these from its layout on its own.
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        if self._playable:
            # A manual move pauses playback. Connected here rather than left to each host: the
            # signal exists for exactly this, and three hosts had already written the same slot.
            self.scrubbed.connect(self._pause_for_manual)

    # -- playback ------------------------------------------------------------ #
    def _build_playback_controls(self) -> None:
        """Play/pause, a speed menu (negative plays backward), and the note that tells the truth
        about the rate."""
        self._play_btn = QPushButton("Play")
        self._play_btn.setObjectName("scrubber_play")
        self._play_btn.setCheckable(True)
        self._play_btn.setFocusPolicy(
            Qt.FocusPolicy.NoFocus
        )  # keep arrows on the scrubber
        self._play_btn.toggled.connect(self._toggle_play)
        self._controls.addWidget(self._play_btn)

        self._speed_box = QComboBox()
        self._speed_box.setObjectName("scrubber_speed")
        self._speed_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for speed in PLAYBACK_SPEEDS:
            self._speed_box.addItem(f"{speed:g}x", speed)
        self._speed_box.setCurrentIndex(PLAYBACK_SPEEDS.index(1.0))
        self._speed_box.currentIndexChanged.connect(self._on_speed_changed)
        self._controls.addWidget(self._speed_box)

        # Blank unless the decode cannot keep up. A rate the surface is not achieving is the one
        # thing a player must not claim: an 8 fps picture under a "1x" label is what makes people
        # distrust the preview rather than the machine.
        #
        # **It holds its width whether it is saying anything or not.** Appearing and disappearing
        # between the speed menu and the step buttons shoved those buttons ~160px sideways every
        # time the decode fell behind and back again — so the control you were reaching for moved
        # while you reached for it. Reserved from the widest thing it can say, in font metrics
        # rather than pixels, so it follows a user who has asked for larger text.
        #
        # And it states the rate rather than naming a fault. "decode-limited" reads as something
        # to fix, and there is nothing here for a user to fix: the video is as heavy as it is.
        # The number is the honest part; the tooltip carries the explanation for anyone who wants
        # it, which is where an explanation nobody has to act on belongs.
        self._rate_note = QLabel("")
        self._rate_note.setEnabled(False)
        self._rate_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rate_note.setFixedWidth(text_width(_RATE_NOTE_WIDEST))
        self._rate_note.setToolTip(
            "The rate this video is actually decoding at. It is below the speed you picked, "
            "which is a property of the footage rather than something to fix."
        )
        self._controls.addWidget(self._rate_note)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance)
        self._apply_interval()

    def set_playback_rate(self, fps: float) -> None:
        """The surface's own frame rate, so 1x means what it says.

        Supplied by the host because only the host knows: a recording's median timestamp step, a
        tool manifest's probed fps. The widget never guesses one — a surface whose index has no
        wall-clock meaning is ``playable=False`` instead.
        """
        self._fps = float(fps) if fps and fps > 0 else DEFAULT_PLAYBACK_FPS
        self._apply_interval()

    def is_playing(self) -> bool:
        return bool(self._play_timer is not None and self._play_timer.isActive())

    def _direction(self) -> int:
        """``1`` forward, ``-1`` backward — the sign of the chosen speed."""
        return -1 if self._speed < 0 else 1

    def playing_direction(self) -> int:
        """``1`` playing forward, ``-1`` backward, ``0`` stopped (or armed and not yet running)."""
        return self._direction() if self.is_playing() else 0

    def set_speed(self, speed: float) -> None:
        """Choose a speed from the menu, as a person would; negative plays backward.

        Raises:
            ValueError: ``speed`` is not one the menu offers.
        """
        if self._speed_box is None:
            raise ValueError("this surface does not play")
        index = self._speed_box.findData(float(speed))
        if index < 0:
            raise ValueError(
                f"{speed}x is not in the speed menu {self.playback_speeds()}"
            )
        self._speed_box.setCurrentIndex(index)

    def play(self) -> None:
        if self._play_btn is not None:
            self._play_btn.setChecked(True)

    def pause(self) -> None:
        if self._play_btn is not None:
            self._play_btn.setChecked(False)

    def _apply_interval(self) -> None:
        if self._play_timer is None:
            return
        # Floored at 8 ms: Qt cannot honour a shorter one, and pretending otherwise would make the
        # achieved-rate note fire on every fast source.
        self._play_timer.setInterval(
            max(int(round(1000.0 / (self._fps * abs(self._speed)))), 8)
        )

    def _on_speed_changed(self, _index: int) -> None:
        speed = self._speed_box.currentData()
        if speed is None:
            return  # no current item — an empty menu has no speed, and -1 is not 1x
        self._speed = float(speed)
        self._render_times.clear()  # the old samples were measured against another target
        self._apply_interval()
        if self.is_playing():
            self._seat_for_direction()  # turned round at an end: start from the other one
        self._refresh_rate_note()

    def set_playback_range(self, first: int | None, last: int | None) -> None:
        """Play only between ``first`` and ``last`` inclusive; ``None`` for the whole span.

        Scrubbing is unaffected — the slider still reaches every frame. This is only where
        *playing* starts and stops, which is the distinction a trim window needs: the scrubber
        spans a padded window so the clip's edges can be dragged, and playing the padding is not
        what the person pressed play for.
        """
        self._play_first = None if first is None else int(first)
        self._play_last = None if last is None else int(last)

    def playback_range(self) -> tuple[int, int]:
        """The resolved frames playback runs between, clamped to what exists."""
        last_frame = max(self._total - 1, 0)
        first = 0 if self._play_first is None else self._play_first
        last = last_frame if self._play_last is None else self._play_last
        first = max(0, min(first, last_frame))
        last = max(first, min(last, last_frame))
        return first, last

    def _pause_for_manual(self) -> None:
        """A user-driven move stops playback; ``_advance`` uses ``_apply`` and never lands here."""
        self._start_on_first_frame = False
        self.pause()

    def _seat_for_direction(self) -> None:
        """Put the playhead where playing in the current direction starts, if it is not inside.

        Forward from the last frame, or backward from the first, would stop on its first tick;
        pressed there, the person meant "again", so it starts over from the far end.
        """
        first, last = self.playback_range()
        if self._direction() > 0 and (self._frame >= last or self._frame < first):
            self._apply(first, decode=True)
        elif self._direction() < 0 and (self._frame <= first or self._frame > last):
            self._apply(last, decode=True)

    def _toggle_play(self, on: bool) -> None:
        if self._play_btn is None:
            return
        self._play_btn.setText("Pause" if on else "Play")
        if not on:
            self._play_timer.stop()
            self._start_on_first_frame = False
            self._refresh_rate_note()
            return
        first, last = self.playback_range()
        if self._total <= 1 or last <= first:
            self._play_btn.setChecked(False)  # nothing to play
            return
        self._seat_for_direction()
        self._render_times.clear()
        if self._has_rendered:
            self._play_timer.start()
        else:
            # Arm rather than start. The clock begins when a frame is actually on screen —
            # otherwise a slow first decode is counted as playback nobody saw, and an
            # undecodable source runs silently to the end.
            self._start_on_first_frame = True
            self._request(self._frame)

    def _advance(self) -> None:
        """One playback tick, within the playback range, looping only if this surface loops.

        Backward is the mirror: it stops on the first frame, or loops round to the last.

        Goes through ``_apply``, not ``_step``: ``_step`` emits ``scrubbed``, which is wired to
        ``_pause_for_manual``, so a player built on it would stop itself on its first tick.
        """
        first, last = self.playback_range()
        direction = self._direction()
        nxt = self._frame + direction
        if nxt > last or nxt < first:
            if self._loop and last > first:
                self._apply(first if direction > 0 else last, decode=True)
            else:
                self.pause()  # rest on the frame playback ran to
            return
        self._apply(nxt, decode=True)

    def _note_rendered_frame(self) -> None:
        self._render_times.append(time.monotonic())
        self._refresh_rate_note()

    def achieved_fps(self) -> float | None:
        """Frames actually put on screen per second, or ``None`` before there are enough."""
        if len(self._render_times) < 2:
            return None
        span = self._render_times[-1] - self._render_times[0]
        return (len(self._render_times) - 1) / span if span > 0 else None

    def _refresh_rate_note(self) -> None:
        if self._rate_note is None:
            return
        achieved = self.achieved_fps()
        wanted = self._fps * abs(self._speed)
        if (
            not self.is_playing()
            or achieved is None
            or achieved >= wanted * _RATE_HONESTY
        ):
            self._rate_note.setText("")
            return
        self._rate_note.setText(f"{achieved:.0f} fps")

    # -- what a structural rule asks ----------------------------------------- #
    # Named accessors rather than letting the rules reach into `_play_btn` and friends. A rule
    # that reads a private attribute breaks on a rename with no signal to whoever renamed it,
    # so these five are part of the widget's contract rather than an accident of its layout.
    def is_playable(self) -> bool:
        """Whether this surface was built to play, which is a decision each host makes."""
        return self._playable

    def play_button(self) -> QPushButton | None:
        """The one control that starts and stops playback, or ``None`` on a still surface."""
        return self._play_btn

    def playback_speeds(self) -> tuple[float, ...]:
        """The rates the speed menu offers, read off the menu rather than off the constant.

        Off the menu on purpose: the constant says what was intended and the menu is what a
        person can actually pick, and a rule about an offer has to check the offer.
        """
        if self._speed_box is None:
            return ()
        return tuple(
            self._speed_box.itemData(i) for i in range(self._speed_box.count())
        )

    def time_labels(self) -> tuple[QLabel, QLabel]:
        """``(elapsed, total)`` either side of the slider — hidden when the host gave no times."""
        return self._elapsed, self._total_time

    def rate_note(self) -> QLabel | None:
        """The label that carries the achieved rate when the decode cannot keep up."""
        return self._rate_note

    def step_buttons(self) -> tuple[QPushButton, ...]:
        """Back one frame and forward one frame — the pair that survives whatever play does.

        Read off the **control row**, not off ``self._prev`` / ``self._nxt``. A button held by an
        attribute is a button that exists; a button in the row is one a person can press, and the
        failure worth catching is the second. Holding the reference while dropping the widget from
        the layout leaves the attribute answering yes about something nobody can see — which is
        the same "derive the claim from the wiring" rule the shared-widget cues follow.
        """
        names = ("scrubber_prev_frame", "scrubber_next_frame")
        found = []
        for i in range(self._controls.count()):
            widget = self._controls.itemAt(i).widget()
            if widget is not None and widget.objectName() in names:
                found.append(widget)
        return tuple(found)

    # -- host API ------------------------------------------------------------ #
    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt virtual
        """The control row wraps, so this widget's height depends on its width."""
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt virtual
        """Delegated to the layout, which is the thing that knows how the row wraps.

        ``QWidget`` does **not** do this for you: the default returns -1 whatever the layout
        says, so a host asking a scrubber how tall it needs to be at a given width got no
        answer and fell back to a one-line ``sizeHint`` — which is the state in which a wrapped
        second line is drawn outside the widget and clipped away.
        """
        return self.layout().heightForWidth(width)

    def add_control(self, widget: QWidget) -> None:
        """Insert a host widget (e.g. a checkbox) into the control row, after the step buttons."""
        self._controls.insertWidget(self._controls.indexOf(self._nxt) + 1, widget)

    def set_transport_compact(self, on: bool) -> None:
        """Play/Pause, the speed menu and any added controls only: the slider, its time
        labels, the step buttons and the frame readout are hidden. For a surface whose frames
        are not one timeline — the clip montage, where every tile loops on its own, so a shared
        playhead and a frame step would point at nothing. Playback itself is unchanged.
        """
        self._compact = bool(on)  # `_refresh_readout` keeps the time labels hidden
        for widget in (
            self._slider,
            self._elapsed,
            self._total_time,
            self._prev,
            self._nxt,
            self._readout,
        ):
            widget.setVisible(not on)

    def set_loop(self, on: bool) -> None:
        """Replay from the start of the playback range instead of stopping at its end."""
        self._loop = bool(on)

    def set_underlay(self, widget: QWidget) -> None:
        """Place a widget directly beneath the slider — a strip describing the frames the
        slider scrubs over (e.g. where a track has embeddings). Use ``handle_travel`` to
        line its drawing up with the frame the handle points at."""
        self.layout().insertWidget(1, widget)

    def handle_travel(self) -> tuple[int, int]:
        """``(left, width)`` in pixels of the slider handle's centre travel.

        An underlay that maps frames to x by its own width is off by half a handle at each
        end — the groove is inset and the handle's centre never reaches the widget edge. A
        strip that mis-points by even a few pixels is worse than none, so it must measure
        the real geometry rather than assume the slider fills its widget. The slider's own
        marks are drawn from the same measurement.
        """
        left, width = self._slider.travel()
        # In this widget's coordinates: the elapsed time sits left of the slider, so an underlay
        # spanning the widget has to start counting where the slider does.
        return self._slider.x() + left, width

    def set_marks(
        self,
        *,
        ticks: Sequence[int] = (),
        spans: Sequence[tuple[int, int]] = (),
        bands: Sequence[tuple[int, int, str | None, bool]] = (),
    ) -> None:
        """Draw ``ticks`` (scrub positions), ``spans`` (inclusive ``(first, last)`` pairs) and
        ``bands`` on the slider, replacing whatever was there.

        A span is *the* clip inside its context: bracketed, with the rest dimmed. A band is one
        of many marked ranges, ``(first, last, color, hollow)`` — a theme colour (``None`` for
        the default) drawn under the groove, outlined when ``hollow``. The example dialog draws
        each behavior's clips as bands in its colour, a non-example hollow.

        Generic on purpose — a pending clip start is a tick, a clip inside its context is a span —
        so a surface that needs to point at part of the timeline asks this rather than growing a
        strip of its own. Cleared by ``set_total``: marks describe the content they were set on.
        """
        self._slider.set_marks(ticks, spans, bands)

    def bands(self) -> tuple[tuple[int, int, str | None, bool], ...]:
        """The bands as drawn (see :meth:`set_marks`)."""
        return self._slider.bands()

    def clear_marks(self) -> None:
        self._slider.set_marks((), ())

    def marks(self) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        """``(ticks, spans)`` as drawn — ask this rather than the slider behind it."""
        return self._slider.marks()

    def set_timestamps(self, t_ns: np.ndarray | Sequence[int] | None) -> None:
        """The recording time of each scrub position, from the timestamp map (Tenet 8).

        One ``t_ns`` per position, so after ``set_total``. ``None`` hides the time: the widget
        never derives one from a frame rate, because frame ÷ fps drifts on any file whose header
        rate differs from its real one, and the map is the only time the pipeline trusts.

        Raises:
            ValueError: ``t_ns`` does not have one entry per scrub position.
        """
        if t_ns is None:
            self._timestamps = None
        else:
            array = np.asarray(t_ns, dtype=np.int64)
            if array.ndim != 1 or len(array) != self._total:
                raise ValueError(
                    f"set_timestamps needs one t_ns per scrub position: got "
                    f"{array.shape} for {self._total} positions — call it after set_total"
                )
            self._timestamps = array
        self._refresh_readout()

    def time_text(self) -> str | None:
        """What the time readout says, or ``None`` when there is no time to say."""
        if self._timestamps is None or not len(self._timestamps):
            return None
        return format_time(int(self._timestamps[self._clamp(self._frame)]))

    def set_decoder(self, compute_fn: Callable[[int], object]) -> None:
        """Register the off-thread frame compute. ``compute_fn(frame)`` runs on a
        worker thread (must do no Qt); its return value arrives via ``rendered``."""
        self._compute_fn = compute_fn

    def set_total(self, total: int) -> None:
        self._total = max(int(total), 0)
        self._slider.setRange(0, max(self._total - 1, 0))
        self._frame = min(self._frame, max(self._total - 1, 0))
        # New content: stop, go back to arming on its first frame, and forget the window the
        # *old* content was played inside. A host that swaps what the scrubber is over (another
        # clip, another recording) has not asked for any of it to carry — and a stale range is
        # the quiet one: it clamps playback to frames that describe something else, so pressing
        # Play does nothing and no error says why. A host with a window re-states it after
        # `set_total`, which is where it knows the new content's shape.
        self._play_first = None
        self._play_last = None
        # Marks and times belong to the old content too; a host re-states them after this.
        self._slider.set_marks((), ())
        self._timestamps = None
        if self._playable:
            self.pause()
            self._has_rendered = False
        self._refresh_readout()

    def set_frame(self, frame: int, *, decode: bool = True) -> None:
        """Move to ``frame`` programmatically (strip click, reload, initial show)."""
        self._apply(frame, decode=decode)

    def frame(self) -> int:
        return self._frame

    @property
    def big_jump(self) -> int:
        return max(self._total // 100, 10)

    # -- readout ------------------------------------------------------------- #
    def _refresh_readout(self) -> None:
        last = max(self._total - 1, 0)
        time_text = self.time_text()
        shown = time_text is not None and not getattr(self, "_compact", False)
        self._elapsed.setVisible(shown)
        self._total_time.setVisible(shown)
        self._elapsed.setText(time_text or "")
        self._total_time.setText(
            format_time(int(self._timestamps[-1])) if shown else ""
        )
        if self._entry is not None:
            self._entry.validator().setTop(last)
            if not self._entry.hasFocus():  # don't clobber what the user is typing
                self._entry.setText(str(self._frame))
            self._max_label.setText(f"/ {last}")
        else:
            self._label.setText(self._label_formatter(self._frame, last))

    def _on_entry(self) -> None:
        text = self._entry.text().strip()
        if text:
            self.scrubbed.emit()  # manual jump — pause autoplay
            self.set_frame(int(text))  # validator → digits only; _apply clamps
        else:
            self._refresh_readout()  # revert empty input
        self._slider.setFocus()  # resume arrow-stepping

    # -- frame change + background decode ------------------------------------ #
    def _clamp(self, frame: int) -> int:
        return min(max(int(frame), 0), max(self._total - 1, 0))

    def _apply(self, frame: int, *, decode: bool) -> None:
        frame = self._clamp(frame)
        self._frame = frame
        self._slider.blockSignals(True)  # avoid re-entering via valueChanged
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._refresh_readout()
        self.frame_changed.emit(frame)  # immediate, cheap
        if decode:
            self._request(frame)

    def _request(self, frame: int) -> None:
        self._pending = frame
        if self._compute_fn is not None and not self._busy:
            self._dispatch()

    def _dispatch(self) -> None:
        frame = self._pending
        self._pending = None
        self._busy = True
        self.busy_changed.emit(True)
        QThreadPool.globalInstance().start(
            _DecodeTask(frame, self._compute_fn, self._signals)
        )

    def _on_decoded(self, frame: int, payload: object) -> None:
        self._busy = False
        if not isinstance(payload, _DecodeFailed):
            self._has_rendered = True
            self.rendered.emit(frame, payload)
            if self._playable:
                self._note_rendered_frame()
                if self._start_on_first_frame:
                    self._start_on_first_frame = False
                    self._play_timer.start()  # the clock starts with the first real frame
        elif self._start_on_first_frame:
            # Armed against a source that cannot decode: disarm rather than run to the end
            # showing nothing.
            self._start_on_first_frame = False
            self.pause()
        if self._pending is not None:  # a newer frame arrived mid-decode → latest only
            self._dispatch()
        else:
            self.busy_changed.emit(False)  # caught up — clear the "decoding…" state

    def _reset_button_hold(self) -> None:
        self._btn_held = 0

    def _button_step(self, direction: int) -> None:
        """One ◀/▶ press (or auto-repeat tick) — accelerates the longer it is held, up to
        ``HOLD_MAX_SPEED``."""
        self._step(direction * self._held_step(self._btn_held))
        self._btn_held += 1
        self._btn_idle.start()  # restart; only fires once the auto-repeat ticks stop

    def _held_step(self, held: int) -> int:
        now = time.monotonic()
        since = (
            None
            if held == 0 or self._last_step_at is None
            else now - self._last_step_at
        )
        self._last_step_at = now
        return held_step(held, self._fps, since)

    def _step(self, delta: int) -> None:
        self.scrubbed.emit()  # manual — a host player pauses autoplay on this
        self._apply(self._frame + delta, decode=True)

    def _on_slider(self, value: int) -> None:
        # user-driven (mouse drag / wheel); programmatic moves block signals
        self.scrubbed.emit()
        self._apply(value, decode=True)

    # -- keyboard (accelerating arrow stepping) ------------------------------ #
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        # The C++ slider can outlive this Python object: the filter stays installed while the
        # scrubber's attributes are already gone, so an event arriving in that window raises
        # `AttributeError: no attribute '_slider'` out of a slot. Qt reports it and carries
        # on, which in the app is a dialog nobody can act on and in the suite was invisible
        # until `tests/gui/conftest.py` started failing on a swallowed slot exception.
        if getattr(self, "_slider", None) is None:
            return False
        if obj is self._slider and event.type() == QEvent.Type.KeyPress:
            if self._handle_key(event):
                return True
        if obj is self._slider and event.type() == QEvent.Type.KeyRelease:
            if not event.isAutoRepeat() and event.key() in (
                Qt.Key.Key_Left,
                Qt.Key.Key_Right,
            ):
                self._held = 0
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        if not self._handle_key(event):
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if not event.isAutoRepeat() and event.key() in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
        ):
            self._held = 0
        super().keyReleaseEvent(event)

    def _handle_key(self, event) -> bool:
        key = event.key()
        shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            direction = -1 if key == Qt.Key.Key_Left else 1
            if shift:
                step = self.big_jump
            else:
                self._held = self._held + 1 if event.isAutoRepeat() else 0
                step = self._held_step(self._held)
            self._step(direction * step)
            return True
        if key == Qt.Key.Key_PageDown:
            self._step(-self.big_jump)
            return True
        if key == Qt.Key.Key_PageUp:
            self._step(self.big_jump)
            return True
        if key == Qt.Key.Key_Home:
            self.scrubbed.emit()
            self._apply(0, decode=True)
            return True
        if key == Qt.Key.Key_End:
            self.scrubbed.emit()
            self._apply(max(self._total - 1, 0), decode=True)
            return True
        if key == Qt.Key.Key_Space and self._play_btn is not None:
            # Not routed through `scrubbed`: that signal means a *manual move*, which is wired
            # to pause, so a space handled there would stop playback in the act of starting it.
            self._play_btn.toggle()
            return True
        return False
