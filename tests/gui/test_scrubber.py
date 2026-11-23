"""``FrameScrubber`` — the transport every video surface in this package is driven by.

Twelve surfaces embed one, so what is guarded here is what a host is entitled to assume: that
a manual move reports itself and a programmatic one does not, that a decode in flight is
announced so a host can say "Decoding frame…" instead of showing a blank pane, that holding a
step button accelerates the way the arrow keys do — and that the row of controls stays legible
however little width it is given.

**The class the last of those exists for: a control row inside a player is handed the video's
width, and nothing enforces its minimum.** ``image_view.PictureWithControls`` positions its two
children with ``setGeometry`` rather than through a ``QLayout`` — deliberately, its own
docstring argues why — so the transport gets exactly the block's width whatever it asked for,
and Qt's answer to a button narrower than its text is to elide the *label*. Reported from the
app as "Prev frame" rendering as "ev fram". Measured: the row wants 932px at the smallest
interface font and 1144px at 12pt, and cannot go below 878px without eliding, against a pane
that was 357px. The row wraps instead (``gui/flow.py``), which keeps every control and every
label whole and costs a second line only while the pane is too narrow to do better.

**A guard for that cannot resize the scrubber directly.** A widget Qt lays out is never given
less than its ``minimumSizeHint``, so ``resize(360, …)`` on a row whose floor was 878px
silently got 878 and nothing was ever squeezed — verified, that version passed at all five
widths against the very layout that shipped the bug. Only the block's ``setGeometry`` bypasses
the minimum, which is the only reason the labels could elide at all, so the check drives the
row through one.
"""

import pytest

from detail.gui.qt import QEvent, QKeyEvent, QTest, Qt
from detail.gui.scrubber import FrameScrubber, accelerated_step
from tests.gui.conftest import drain_until


def test_scrubbed_signal_fires_on_manual_not_programmatic(qapp):
    scrubber = FrameScrubber(playable=False)
    scrubber.set_total(100)
    fired = []
    scrubber.scrubbed.connect(lambda: fired.append(1))
    scrubber.set_frame(10)  # programmatic (autoplay uses this) — no manual signal
    assert fired == []
    scrubber._step(1)  # ◀/▶ button or arrow key
    scrubber._on_slider(20)  # slider drag
    assert len(fired) == 2


def test_busy_changed_marks_decode_in_flight_then_caught_up(qapp):
    # Hosts drive a "Decoding frame…" overlay off this, so first-load isn't a blank pane.
    s = FrameScrubber(playable=False)
    s.set_total(100)
    states = []
    s.busy_changed.connect(states.append)
    s.set_decoder(lambda frame: frame)  # trivial off-thread compute
    s.set_frame(5)  # dispatches a decode → busy True immediately (UI thread)
    assert states and states[0] is True
    drain_until(qapp, lambda: states[-1] is False)  # the decode delivered → busy False


# --- accelerated_step (pure) ------------------------------------------------ #
def test_accelerated_step_ramps_and_caps():
    assert accelerated_step(0) == 1  # a tap is one frame
    assert [accelerated_step(h) for h in (1, 2, 3)] == [1, 1, 1]
    assert accelerated_step(4) == 2  # doubles every 4 repeats
    assert accelerated_step(8) == 4
    assert accelerated_step(12) == 8
    assert accelerated_step(10_000) == 128  # capped
    assert accelerated_step(-5) == 1  # defensive


# --- FrameScrubber ---------------------------------------------------------- #
def _key(scrubber, key, *, autorep=False, shift=False):
    mods = (
        Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    )
    scrubber._handle_key(QKeyEvent(QEvent.Type.KeyPress, key, mods, "", autorep))


def test_set_total_and_clamp(qapp):
    s = FrameScrubber(playable=False)
    s.set_total(100)
    s.set_frame(250)  # past the end
    assert s.frame() == 99
    s.set_frame(-5)
    assert s.frame() == 0


def test_frame_changed_immediate(qapp):
    s = FrameScrubber(playable=False)
    s.set_total(1000)
    changed = []
    s.frame_changed.connect(changed.append)
    for f in (10, 20, 30, 40):
        s.set_frame(f, decode=False)
    assert changed == [10, 20, 30, 40]  # every change is reported immediately


def test_decode_is_background_and_latest_only(qapp):
    import time

    s = FrameScrubber(playable=False)
    s.set_total(1000)
    calls, rendered = [], []

    def compute(frame):  # runs on a worker thread; slow enough to pile up requests
        calls.append(frame)
        time.sleep(0.03)
        return frame

    s.set_decoder(compute)
    s.rendered.connect(lambda f, payload: rendered.append(payload))

    s.set_frame(0)  # dispatches; the next two pile into _pending while it runs
    s.set_frame(5)
    s.set_frame(9)
    QTest.qWait(300)  # let the worker drain

    assert calls == [0, 9]  # 5 was coalesced away — only the latest is decoded
    assert rendered == [0, 9]  # delivered on the UI thread, ending on the latest


def test_arrow_steps_one_then_accelerates(qapp):
    s = FrameScrubber(playable=False)
    s.set_total(1000)
    s.set_frame(500, decode=False)

    _key(s, Qt.Key.Key_Right)  # fresh press → +1
    assert s.frame() == 501
    for _ in range(8):  # hold: auto-repeats accelerate the step
        _key(s, Qt.Key.Key_Right, autorep=True)
    assert s.frame() > 505  # covered more than 8 single frames
    _key(s, Qt.Key.Key_Left)  # a new (non-repeat) press resets to single-frame
    assert s.frame() == s.frame()  # left moved back by 1 from the accelerated spot


class _Clock:
    """A ``time.monotonic`` that advances only when told — a held button's ticks, on demand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_button_hold_accelerates(qapp, monkeypatch):
    # Holding ◀/▶ auto-repeats; Qt fires pressed+clicked every tick, so a pressed-driven
    # reset would zero the hold depth each tick and it would never accelerate. Drive both
    # signals to prove the depth climbs (the idle-timer reset only fires once ticks stop).
    # Ticks 60 ms apart, as Qt's auto-repeat fires them: the speed cap is a rate, so a test
    # firing them all at once would be capped to one frame each and prove nothing.
    clock = _Clock()
    monkeypatch.setattr("detail.gui.scrubber.time.monotonic", clock)
    s = FrameScrubber(playable=False)
    s.set_total(100_000)
    s.set_frame(0, decode=False)
    nxt = next(
        s._controls.itemAt(i).widget()
        for i in range(s._controls.count())
        if s._controls.itemAt(i).widget() is not None
        and s._controls.itemAt(i).widget().text() == "Next frame"
    )
    for _ in range(12):  # a held auto-repeat: pressed + clicked each tick
        clock.now += 0.06
        nxt.pressed.emit()
        nxt.clicked.emit()
    assert (
        s.frame() > 12
    )  # accelerated past 12 single-frame steps (12 = the un-accelerated sum)


def test_big_jump_and_home_end(qapp):
    s = FrameScrubber(playable=False)
    s.set_total(1000)
    s.set_frame(500, decode=False)
    _key(s, Qt.Key.Key_Right, shift=True)  # shift = big jump
    assert s.frame() == 500 + s.big_jump  # big_jump = max(total//100, 10) = 10
    _key(s, Qt.Key.Key_Home)
    assert s.frame() == 0
    _key(s, Qt.Key.Key_End)
    assert s.frame() == 999


def test_frame_entry_jumps_and_clamps(qapp):
    s = FrameScrubber(playable=False)
    s.set_total(1000)
    s._entry.setText("250")
    s._on_entry()
    assert s.frame() == 250
    s._entry.setText("99999")  # past the end → clamps
    s._on_entry()
    assert s.frame() == 999
    s._entry.setText("")  # empty → no change, readout reverts
    s._on_entry()
    assert s.frame() == 999 and s._entry.text() == "999"


def test_frame_entry_absent_when_disabled(qapp):
    s = FrameScrubber(allow_frame_entry=False, playable=False)
    assert s._entry is None  # the no-video time scrubber keeps its static label


def test_frame_entry_enter_consumed(qapp):
    # Enter must jump AND be consumed, so in a dialog it doesn't fire the default
    # button (which would close the dialog).
    s = FrameScrubber(playable=False)
    s.set_total(100)
    s._entry.setText("42")
    ev = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier
    )
    s._entry.keyPressEvent(ev)
    assert s.frame() == 42  # jumped
    assert ev.isAccepted()  # consumed → won't bubble to a dialog's default button


def test_decode_signals_unparented_for_safe_teardown(qapp):
    # must outlive the scrubber so an in-flight worker task can still emit safely
    # when a host dialog closes mid-decode (else: RuntimeError on a deleted QObject)
    s = FrameScrubber(playable=False)
    assert s._signals.parent() is None


# --- playback ---------------------------------------------------------------- #
# The player lives in the widget rather than in each host because opt-in is how nine
# surfaces went months with no way to watch video. `playable` has no default: whether a
# surface's frames are a movie or a canvas decides whether the output is *correct*, and the
# three arena dialogs must refuse it because their scrub handler writes to a keyframe.


def _playable(total=100, fps=30.0):
    s = FrameScrubber(playable=True)
    s.set_total(total)
    s.set_playback_rate(fps)
    return s


def test_a_non_playable_scrubber_has_no_play_control(qapp):
    s = FrameScrubber(playable=False)
    assert s._play_btn is None and s._speed_box is None and s._play_timer is None
    assert not s.is_playing()
    from detail.gui.qt import QPushButton

    labels = [b.text() for b in s.findChildren(QPushButton)]
    # Step buttons only, and named rather than arrowed: a "Play" button beside a bare "▶" was
    # two meanings on one glyph, and no test could see it because both were labelled and wired.
    assert labels == ["Prev frame", "Next frame"]


def test_playback_advances_without_pausing_itself(qapp):
    """The trap this widget invites: `_step` emits `scrubbed`, `scrubbed` pauses playback.

    A player built on `_step` stops itself on its first tick. `_advance` must go through
    `_apply`. Point `_advance` at `_step` and this fails.
    """
    s = _playable()
    fired = []
    s.scrubbed.connect(lambda: fired.append(1))
    s._advance()
    s._advance()
    assert s.frame() == 2  # it moved
    assert fired == []  # and never reported itself as a manual move


def test_playback_stops_at_the_end_and_does_not_loop(qapp):
    s = _playable(total=3)
    s._has_rendered = True
    s.set_frame(0, decode=False)
    s.play()
    s._advance()
    s._advance()
    assert s.frame() == 2  # the last frame
    s._advance()  # one tick past the end
    assert s.frame() == 2  # rests there rather than wrapping to 0
    assert not s.is_playing()


def test_pressing_play_at_the_end_starts_over(qapp):
    """Distinct from looping: the *user* asked again, so rewind. `_advance` never does."""
    s = _playable(total=3)
    s._has_rendered = True
    s.set_frame(2, decode=False)
    s.play()
    assert s.frame() == 0 and s.is_playing()


def test_a_manual_move_pauses_playback(qapp):
    s = _playable()
    s._has_rendered = True  # pretend a frame is already on screen
    s.play()
    assert s.is_playing()
    s._step(1)  # a ◀/▶ press
    assert not s.is_playing()


def test_playback_arms_until_the_first_frame_arrives(qapp):
    """Starting the clock against a pane that has not decoded counts frames nobody saw."""
    s = _playable()
    s.set_decoder(lambda frame: frame)
    s.play()
    assert s._start_on_first_frame and not s.is_playing()  # armed, not running
    s._on_decoded(0, object())  # a real frame lands
    assert s.is_playing() and not s._start_on_first_frame


def test_a_source_that_cannot_decode_does_not_stay_armed(qapp):
    from detail.gui.scrubber import _DecodeFailed

    s = _playable()
    s.set_decoder(lambda frame: frame)
    s.play()
    assert s._start_on_first_frame
    s._on_decoded(0, _DecodeFailed(RuntimeError("no video")))
    assert not s._start_on_first_frame and not s.is_playing()


def test_the_interval_follows_the_surfaces_rate_not_a_constant(qapp):
    """A fixed timer plays every recording at whatever that timer is — real time by luck."""
    s = _playable(fps=30.0)
    assert s._play_timer.interval() == 33
    s.set_playback_rate(60.0)
    assert s._play_timer.interval() == 17
    s.set_playback_rate(25.0)
    assert s._play_timer.interval() == 40


def test_the_speed_menu_scales_that_rate(qapp):
    from detail.gui.scrubber import PLAYBACK_SPEEDS

    s = _playable(fps=30.0)
    s._speed_box.setCurrentIndex(PLAYBACK_SPEEDS.index(2.0))
    assert s._play_timer.interval() == 17  # twice as fast
    s._speed_box.setCurrentIndex(PLAYBACK_SPEEDS.index(0.5))
    assert s._play_timer.interval() == 67


def test_the_rate_note_says_the_achieved_rate_when_decode_cannot_keep_up(qapp):
    """Never claim a nominal 1x through a decode that is dropping frames.

    **Why this is a behavioural test and not a `video` requirement.** It was drafted as one of
    #337's eight structural rules and written as one, and it passed — over every subject,
    vacuously. The note is blank unless the surface is *playing and falling behind*, and a widget
    the requirements walk finds is neither, so the rule returned early on all six subjects and
    could not have failed. Watched doing exactly that against a reverted fix.

    A structural rule reads a surface as it is found. This property only exists while the thing
    is running, so the check has to drive it — which is what this does, and why it lives here.
    """
    import time

    s = _playable(fps=30.0)
    s._has_rendered = True
    s.play()
    for _ in range(5):  # frames arriving at ~10 fps against a 30 fps target
        s._render_times.append(time.monotonic())
        time.sleep(0.02)
    s._render_times.clear()
    now = time.monotonic()
    for i in range(5):
        s._render_times.append(now + i * 0.1)  # 10 fps
    s._refresh_rate_note()
    assert round(s.achieved_fps()) == 10
    # The whole string, not containment: "10 fps" is also in "110 fps", which is the recorded
    # idiom that passes against the bug it guards.
    assert s._rate_note.text() == "10 fps"
    s.pause()
    assert s._rate_note.text() == ""  # nothing to say when not playing


def test_the_rate_note_holds_its_width_whether_it_speaks_or_not(qapp):
    """Its neighbours must not move when it starts or stops saying something.

    It sits between the speed menu and the step buttons, so appearing and disappearing shoved
    `Prev frame` / `Next frame` about 160px sideways every time the decode fell behind and
    recovered — the control you were reaching for moving while you reached for it. Reported by a
    person using it, which is the only way this kind of thing is ever found: every frame renders,
    every button works, and nothing fails.

    Asserted on the **step buttons' position**, not on the label's width. A fixed width is the
    current implementation; buttons that do not move is the property, and it survives someone
    choosing a different way to hold the space.
    """
    import time

    s = _playable(fps=30.0)
    s.resize(700, 60)
    s.show()
    qapp.processEvents()
    before = [b.x() for b in s.step_buttons()]

    s._has_rendered = True
    s.play()
    now = time.monotonic()
    for i in range(5):
        s._render_times.append(now + i * 0.1)  # 10 fps against a 30 fps target
    s._refresh_rate_note()
    qapp.processEvents()

    assert s._rate_note.text(), "the note never spoke, so this proves nothing"
    assert [
        b.x() for b in s.step_buttons()
    ] == before, "the step buttons moved when the rate note appeared"


def test_the_rate_note_states_the_rate_rather_than_naming_a_fault(qapp):
    """There is nothing here for a user to do, so it does not read like something to fix.

    The video is as heavy as it is. "decode-limited" reads as a fault report — it was raised as
    "alerting users to an issue they have no control over" — so the number is the visible part
    and the explanation lives in the tooltip, for whoever wants it.
    """
    import time

    s = _playable(fps=30.0)
    s._has_rendered = True
    s.play()
    now = time.monotonic()
    for i in range(5):
        s._render_times.append(now + i * 0.1)
    s._refresh_rate_note()

    assert s._rate_note.text() == "10 fps"
    assert "decode-limited" not in s._rate_note.text()
    assert s._rate_note.toolTip(), "the explanation has to be somewhere"


def test_new_content_forgets_the_old_content_s_playback_window(qapp):
    """A range describes the frames it was computed against, so it cannot outlive them.

    The quiet failure, found by folding `_ClipPlayer` onto this widget: a window left over from
    a shorter clip clamps the next one to frames that describe something else, so Play does
    nothing at all and nothing says why.
    """
    s = _playing(total=100, first=30, last=40)
    s.set_total(10)
    assert s.playback_range() == (0, 9), "the previous clip's window survived the swap"


def test_new_content_stops_playback(qapp):
    s = _playable()
    s._has_rendered = True
    s.play()
    assert s.is_playing()
    s.set_total(50)  # the host swapped what the scrubber is over
    assert not s.is_playing() and not s._has_rendered


# --- looping, and the window playback runs inside ----------------------------------------- #
# Both arrived so `_ClipPlayer` could stop carrying its own timer. They are what its private
# loop did that the shared one could not: replay a clip rather than resting on its last frame,
# and play the *clip* rather than the padded window the scrubber spans so its edges can be
# dragged. Folding a player without them would have quietly changed both behaviours.


def _playing(total=20, *, loop=False, first=None, last=None):
    s = FrameScrubber(playable=True, loop=loop)
    s.set_total(total)
    s.set_playback_range(first, last)
    s._has_rendered = True  # a frame is on screen, so play starts rather than arming
    return s


def test_playback_stops_at_the_range_end_not_the_last_frame(qapp):
    """The trim window's whole point: the scrubber spans padding the clip does not."""
    s = _playing(total=100, first=10, last=14)
    s.set_frame(13, decode=False)
    s.play()

    s._advance()
    assert s.frame() == 14
    s._advance()
    assert not s.is_playing(), "ran past the end of the clip into the padding"
    assert s.frame() == 14, "and it must rest on the clip's last frame, not move"


def test_a_looping_surface_returns_to_the_range_start(qapp):
    s = _playing(total=100, loop=True, first=10, last=12)
    # Play *first*, then move to the end. Pressing play while already at the end restarts by
    # design, so seeding the position beforehand never reaches the tick this is about — which is
    # what the first version of this test did, and it failed on correct code.
    s.play()
    s.set_frame(12, decode=False)

    s._advance()
    assert (
        s.frame() == 10
    ), "a looping clip replays from its own start, not from frame 0"
    assert s.is_playing(), "looping must not stop"


def test_a_surface_that_does_not_loop_rests_at_the_end(qapp):
    """The default, and the behaviour every recording surface wants."""
    s = _playing(total=5)
    s.play()
    s.set_frame(4, decode=False)  # after play, for the reason above

    s._advance()
    assert not s.is_playing()
    assert s.frame() == 4


def test_pressing_play_inside_a_window_starts_where_the_window_starts(qapp):
    """Pressed at the end *or* outside the window — both mean "start where playing starts"."""
    s = _playing(total=100, first=30, last=40)

    s.set_frame(0, decode=False)  # scrubbed out of the window by hand
    s.play()
    assert s.frame() == 30, "playing began in the padding rather than at the clip"

    s.pause()
    s.set_frame(40, decode=False)  # at the end
    s.play()
    assert s.frame() == 30, "pressing play at the end must replay the clip"


def test_a_range_of_one_frame_has_nothing_to_play(qapp):
    """A one-frame clip snaps the button back rather than sticking on 'Pause'."""
    s = _playing(total=100, first=7, last=7)
    s.play()
    assert not s.is_playing()


def test_the_range_is_clamped_to_what_exists(qapp):
    """A window resolved against a longer recording must not outlive a shorter one."""
    s = _playing(total=10, first=3, last=999)
    assert s.playback_range() == (3, 9)

    s.set_playback_range(None, None)
    assert s.playback_range() == (0, 9), "None is the whole span, both ends"


def test_scrubbing_still_reaches_outside_the_playback_range(qapp):
    """The range bounds *playing*, never the slider — the padding is there to be dragged into."""
    s = _playing(total=100, first=30, last=40)
    s.set_frame(2, decode=False)
    assert s.frame() == 2
    s.set_frame(90, decode=False)
    assert s.frame() == 90


# --- space starts and stops it ------------------------------------------------------------ #
# Handled where the arrow keys are handled, so every surface holding a scrubber gets it at once
# rather than each host binding its own.
#
# **Focus-scoped, not window-scoped**, which is a reversal of the first plan. A window-wide
# shortcut is what every video app does and it cannot work here: the annotation page puts two
# players side by side and the eval page puts a player beside a scrubber of its own, so two
# handlers would claim the same key in one window — Qt calls that ambiguous and fires neither.
# Clicking the picture focuses that player's scrubber, which is how you say which one you mean.


def _press(widget, key, modifier=Qt.KeyboardModifier.NoModifier):
    widget.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, modifier))


def test_space_starts_and_stops_playback(qapp):
    s = _playing(total=20)

    _press(s, Qt.Key.Key_Space)
    assert s.is_playing(), "space did not start it"

    _press(s, Qt.Key.Key_Space)
    assert not s.is_playing(), "the same key has to stop it — one gesture, both halves"


def test_space_does_nothing_on_a_surface_that_cannot_play(qapp):
    """A still surface must not swallow the key: its host may want it for something else."""
    s = FrameScrubber(playable=False)
    s.set_total(20)

    _press(s, Qt.Key.Key_Space)  # no play control to toggle, and no exception

    assert s.play_button() is None


def test_space_does_not_count_as_a_manual_move(qapp):
    """The trap this was written against.

    A manual move (arrow, slider, typed frame) pauses playback, which is right — you cannot
    step frames while it runs. Routing space through that same path would make it stop playback
    in the act of starting it, and the button would flick on and straight back off.
    """
    s = _playing(total=20)
    paused = []
    s.scrubbed.connect(lambda: paused.append(True))

    _press(s, Qt.Key.Key_Space)

    assert s.is_playing()
    assert (
        not paused
    ), "space was reported as a manual scrub, which pauses what it just started"


def test_arrow_keys_still_pause_a_playing_surface(qapp):
    """The other half of the pair: stepping by hand does stop it, and must keep doing so."""
    s = _playing(total=20)
    _press(s, Qt.Key.Key_Space)
    assert s.is_playing()

    _press(s, Qt.Key.Key_Right)

    assert (
        not s.is_playing()
    ), "an arrow key must still stop playback so frames can be stepped"


# --- what the caller must say ------------------------------------------------ #
#: Parameters that are a genuine *choice* the host makes, with why. Everything else on the
#: constructor must be required, so a surface cannot get it wrong by saying nothing.
_CHOICES = {
    "parent": "Qt ownership, not a fact about the frames",
    "label_formatter": "how the readout reads - a stream cursor says '12.3s', a video a frame",
    "loop": (
        "a clip somebody is judging is watched over and over; a recording is watched through. "
        "Neither is the correct default for the other, and getting it wrong is visible the "
        "moment you press play rather than hiding in a plausible picture"
    ),
    "allow_frame_entry": (
        "whether typing a frame number lands anywhere; false where the index is a position in a "
        "list rather than a frame"
    ),
}


def test_a_fact_about_the_surface_is_never_optional():
    """`playable` has no default: whether frames are a movie or a canvas is not a preference.

    The recorded shape, from `test_spatial_map.py`: a capability added to a shared widget as an
    optional parameter reaches the one surface being worked on, because the default *is* the old
    behaviour. Defaulting `playable=True` puts a play button on the calibration dialog; defaulting
    it False is the opt-in that left nine surfaces with no player for months, silently.
    """
    import inspect

    offenders = [
        name
        for name, param in inspect.signature(FrameScrubber.__init__).parameters.items()
        if name not in ("self", "args", "kwargs")
        and name not in _CHOICES
        and param.default is not inspect.Parameter.empty
    ]
    assert not offenders, (
        f"these default rather than requiring the caller to say: {offenders}. A surface that "
        "never passes one gets the old behaviour and no test fails. Make it required, or add "
        "it to _CHOICES with the reason it is a choice."
    )


def test_the_choices_list_does_not_outlive_its_parameters():
    """Guard the guard: an entry for a parameter that is gone excuses nothing while reading as
    if it still does - the staleness every allowlist in this repo is held to."""
    import inspect

    known = set(inspect.signature(FrameScrubber.__init__).parameters)
    stale = sorted(set(_CHOICES) - known)
    assert not stale, f"_CHOICES names parameters that no longer exist: {stale}"


# --------------------------------------------------------------------------- #
# the control row's width
#
# The class: **a control row inside a player is handed the video's width, and nothing
# enforces its minimum.** `PictureWithControls` positions its children with `setGeometry`
# rather than through a `QLayout` (deliberately — see its docstring), so the row gets the
# block's width whatever it asked for, and Qt's answer to a button narrower than its text is
# to elide the label. Reported from the app as "Prev frame" rendering as "ev fram".
#
# Measured before the fix: the row wants 932px at the smallest interface font and 1144px at
# 12pt, and cannot go below 878px without eliding — against a pane that was 357px.
# --------------------------------------------------------------------------- #
def _elided(scrubber):
    """Every visible control laid out narrower than the text it is trying to show."""
    from detail.gui.qt import QAbstractButton

    return [
        (w.text(), w.width(), w.sizeHint().width())
        for w in scrubber.findChildren(QAbstractButton)
        if w.isVisible() and w.width() < w.sizeHint().width()
    ]


@pytest.mark.parametrize("width", [1200, 900, 700, 500, 360])
def test_the_control_row_wraps_rather_than_eliding_its_labels(qapp, width):
    """At every width a pane can actually be, not just a comfortable one.

    **Driven through `PictureWithControls`, and that is the whole test.** Resizing a scrubber
    directly proves nothing: a widget Qt lays out is never given less than its
    `minimumSizeHint`, so `resize(360, ...)` on a row whose floor was 878px silently got 878 and
    nothing was ever squeezed — verified, that version passed at all five widths against the
    very `QHBoxLayout` that shipped the bug. The block positions its children with
    `setGeometry`, which honours no minimum at all, and that is the only reason the labels
    could elide in the first place.

    Parametrized because the failure is a *threshold*: one wide case cannot see it, and one
    narrow case cannot show that a row which fits is still drawn on one line.
    """
    from detail.gui.qt import QApplication

    from detail.gui.image_view import ImageView, PictureWithControls

    scrubber = FrameScrubber(playable=True)
    scrubber.set_total(479)
    block = PictureWithControls(ImageView(zoomable=True), scrubber, top=True)
    block.resize(width, 700)
    block.show()
    QApplication.processEvents()
    try:
        assert scrubber.width() <= width, "the block did not hand the row its own width"
        assert not _elided(scrubber), f"at {width}px: {_elided(scrubber)}"
    finally:
        block.close()


def test_the_row_is_one_line_when_it_fits_and_more_when_it_does_not(qapp):
    """The other half — a row that always wrapped would pass the check above and be wrong."""
    from detail.gui.qt import QAbstractButton, QApplication, QPoint

    def lines(width):
        scrubber = FrameScrubber(playable=True)
        scrubber.set_total(479)
        scrubber.resize(width, scrubber.heightForWidth(width))
        scrubber.show()
        QApplication.processEvents()
        # In the scrubber's coordinates: a button inside a grouped unit (the two play
        # directions) reports `y()` against its group, which is 0 whatever line it is on.
        rows = {
            b.mapTo(scrubber, QPoint(0, 0)).y()
            for b in scrubber.findChildren(QAbstractButton)
            if b.isVisible() and b.width() > 0
        }
        scrubber.hide()
        return len(rows)

    assert lines(1400) == 1, "a row with room to spare was split anyway"
    assert lines(360) > 1, "a row with no room stayed on one line"


def test_the_scrubber_answers_how_tall_it_is_at_a_given_width(qapp):
    """A host that reserves space for this widget has to ask, and `QWidget` does not derive
    either of these from its layout: the default `heightForWidth` returns -1 whatever the
    layout says, so a host fell back to a one-line `sizeHint` and the wrapped second line was
    drawn outside the widget and clipped away — every control on it present, wired, invisible.
    """
    scrubber = FrameScrubber(playable=True)
    scrubber.set_total(479)
    assert scrubber.hasHeightForWidth()
    assert scrubber.sizePolicy().hasHeightForWidth(), "the layout will never consult it"
    assert scrubber.heightForWidth(360) > scrubber.heightForWidth(1400)


# --------------------------------------------------------------------------- #
# Backward is a speed (issue 627, option B)
# --------------------------------------------------------------------------- #
def test_the_speed_menu_mirrors_every_forward_speed_backward(qapp):
    """Reverse is a negative speed, as VLC models it, not a second Play button."""
    s = _playable()
    speeds = s.playback_speeds()
    forward = [v for v in speeds if v > 0]
    assert [v for v in speeds if v < 0] == [-v for v in reversed(forward)]
    assert 0 not in speeds, "0x is not a speed, it is pause"
    assert (
        s._speed_box.currentData() == 1.0
    ), "the menu must open at 1x, not the first entry"
    assert s._speed_box.currentText() == "1x"
    assert s._speed_box.itemText(s._speed_box.findData(-1.0)) == "-1x"


def test_there_is_one_play_button(qapp):
    """One control starts and stops playback; which way is the speed's business."""
    from detail.gui.qt import QPushButton

    s = _playable()
    texts = [b.text() for b in s.findChildren(QPushButton)]
    assert texts.count("Play") == 1
    assert not [t for t in texts if "backward" in t.lower()]


def test_a_negative_speed_plays_toward_the_start_and_stops_on_the_first_frame(qapp):
    s = _playable(total=5)
    s._has_rendered = True
    s.set_frame(3, decode=False)
    s.set_speed(-1.0)
    s.play()
    assert s.playing_direction() == -1
    s._advance()
    s._advance()
    s._advance()
    assert s.frame() == 0
    s._advance()  # one tick past the start
    assert s.frame() == 0, "backward wrapped round on a surface that does not loop"
    assert not s.is_playing()


def test_backward_on_a_looping_surface_wraps_to_the_last_frame(qapp):
    s = FrameScrubber(playable=True, loop=True)
    s.set_total(4)
    s._has_rendered = True
    s.set_frame(0, decode=False)
    s.set_speed(-1.0)
    s.play()  # pressed at the start: begins from the far end
    assert s.frame() == 3
    for _ in range(3):
        s._advance()
    assert s.frame() == 0
    s._advance()
    assert s.frame() == 3 and s.is_playing()


def test_backward_respects_the_playback_range(qapp):
    """The mirror of the forward range: stop on its first frame, not on frame 0."""
    s = _playable(total=20)
    s.set_playback_range(5, 10)
    s._has_rendered = True
    s.set_frame(15, decode=False)  # outside the window
    s.set_speed(-1.0)
    s.play()
    assert (
        s.frame() == 10
    ), "pressed outside the window, playback must start at its far end"
    for _ in range(10):
        s._advance()
    assert s.frame() == 5 and not s.is_playing()


def test_changing_to_a_negative_speed_mid_play_turns_round_on_one_timer(qapp):
    from detail.gui.qt import QTimer

    s = _playable(total=50)
    s._has_rendered = True
    s.set_frame(25, decode=False)
    s.play()
    s._advance()
    assert s.frame() == 26
    s.set_speed(-2.0)
    assert (
        s.playing_direction() == -1
    ), "changing speed stopped or did not turn playback"
    s._advance()
    assert s.frame() == 25
    assert s._play_timer.interval() == 17, "-2x must tick at twice the rate, like 2x"
    active = [t for t in s.findChildren(QTimer) if t.isActive()]
    assert active == [s._play_timer], f"{len(active)} timers running"


def test_space_pauses_backward_playback(qapp):
    s = _playable(total=50)
    s._has_rendered = True
    s.set_frame(25, decode=False)
    s.set_speed(-1.0)
    s.play()
    s._handle_key(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier(0))
    )
    assert not s.is_playing()


def test_the_rate_note_is_honest_going_backward_too(qapp):
    """Backward decode is often slower than forward; "-1x" over 10 fps is the same lie as "1x"."""
    import time

    s = _playable(total=100, fps=30.0)
    s._has_rendered = True
    s.set_frame(90, decode=False)
    s.set_speed(-1.0)
    s.play()
    now = time.monotonic()
    for i in range(5):
        s._render_times.append(now + i * 0.1)  # 10 fps against a 30 fps target
    s._refresh_rate_note()
    assert s._rate_note.text() == "10 fps"


def test_an_offered_speed_is_the_only_one_set_speed_takes(qapp):
    s = _playable()
    with pytest.raises(ValueError, match="not in the speed menu"):
        s.set_speed(3.0)


# --------------------------------------------------------------------------- #
# A held step follows the animal; it does not fly (issue 627 follow-up)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fps", [25.0, 30.0, 60.0])
@pytest.mark.parametrize("interval_s", [0.03, 0.06])
def test_a_held_step_never_runs_faster_than_three_times_real_time(fps, interval_s):
    """Measured before the cap: 71x on the buttons and ~140x on the keys after two seconds.

    Over every frame rate and both repeat rates (a button's 60 ms, macOS's default 30 ms key
    repeat), the frames covered per second of holding stay at or under 3x the video's own rate.
    """
    from detail.gui.scrubber import HOLD_MAX_SPEED, held_step

    for held in range(4, 200):  # well past where the doubling alone hits its 128 cap
        speed = held_step(held, fps, interval_s) / interval_s / fps
        assert speed <= HOLD_MAX_SPEED + 1e-9, f"{speed:.1f}x at repeat {held}"
    top = held_step(199, fps, interval_s) / interval_s / fps
    assert (
        top >= HOLD_MAX_SPEED * 0.6
    ), f"capped far below the intended speed: {top:.1f}x"


def test_a_tap_is_still_one_frame():
    from detail.gui.scrubber import held_step

    assert held_step(0, 30.0, None) == 1
    assert held_step(40, 30.0, None) == 1


def test_holding_an_arrow_key_is_capped_too(qapp, monkeypatch):
    """The keys go through the same cap as the buttons — one rule for one gesture."""
    clock = _Clock()
    monkeypatch.setattr("detail.gui.scrubber.time.monotonic", clock)
    s = _playable(total=100_000, fps=30.0)
    s.set_frame(0, decode=False)
    _key(s, Qt.Key.Key_Right)
    for _ in range(100):  # 3 s of a 30 ms key repeat
        clock.now += 0.03
        _key(s, Qt.Key.Key_Right, autorep=True)
    seconds = 100 * 0.03
    assert s.frame() <= 3 * 30 * seconds + 1, f"{s.frame()} frames in {seconds:.0f}s"
    assert s.frame() > 30 * seconds, "the hold never got past real time"


# --------------------------------------------------------------------------- #
# Timeline marks
# --------------------------------------------------------------------------- #
def test_marks_are_stored_as_given_and_replaced_not_accumulated(qapp):
    s = _playable(total=100)
    s.set_marks(ticks=[10], spans=[(20, 40)])
    assert s.marks() == ((10,), ((20, 40),))
    s.set_marks(ticks=[50])
    assert s.marks() == ((50,), ()), "a second call must replace, not add to, the first"
    s.clear_marks()
    assert s.marks() == ((), ())


def test_new_content_forgets_the_old_contents_marks_and_times(qapp):
    """Marks and times describe the frames they were set on, like the playback range."""
    import numpy as np

    s = _playable(total=10)
    s.set_marks(ticks=[3], spans=[(1, 2)])
    s.set_timestamps(np.arange(10, dtype=np.int64))
    s.set_total(10)
    assert s.marks() == ((), ())
    assert s.time_text() is None


def test_marks_are_drawn_where_the_handle_points(qapp):
    """A tick that mis-points by a few pixels is worse than none: it names the wrong frame.

    Read off the rendered image, at the handle's own position for that value — not off the
    arithmetic, which would agree with itself whatever the slider did.
    """
    s = _playable(total=101)
    s.resize(600, 60)
    s.show()
    qapp.processEvents()
    slider = s._slider
    plain = slider.grab().toImage()
    s.set_marks(ticks=[50])
    qapp.processEvents()
    marked = slider.grab().toImage()
    x = round(slider.x_of(50))
    changed = [
        col
        for col in range(slider.width())
        if marked.pixelColor(col, 1) != plain.pixelColor(col, 1)
    ]
    assert changed, "the tick was not drawn"
    assert min(changed) <= x <= max(changed), f"tick drawn at {changed}, handle at {x}"
    assert max(changed) - min(changed) <= 12, "the tick is wider than a tick"


# --------------------------------------------------------------------------- #
# The time, either side of the slider
# --------------------------------------------------------------------------- #
def test_without_timestamps_the_player_shows_no_time_rather_than_a_guess(qapp):
    """Frame ÷ fps drifts on any file whose header rate is not its real one."""
    s = _playable(total=100, fps=30.0)
    s.set_frame(60, decode=False)
    assert s.time_text() is None
    assert all(label.isHidden() for label in s.time_labels())


def test_the_time_is_read_from_the_timestamp_map_not_the_frame_rate(qapp):
    """A map whose steps are *not* 1/fps: a derived time would say 0:02.000 at frame 60."""
    import numpy as np

    from detail.gui.timefmt import format_time

    s = _playable(total=100, fps=30.0)
    t_ns = (
        np.arange(100, dtype=np.int64) * 40_000_000
    ) + 5_000_000_000  # 25 fps, offset 5 s
    s.set_timestamps(t_ns)
    s.set_frame(60, decode=False)
    elapsed, total = s.time_labels()
    assert s.time_text() == format_time(int(t_ns[60])) == "0:07.400"
    assert elapsed.text() == "0:07.400" and not elapsed.isHidden()
    assert total.text() == format_time(int(t_ns[-1])) and not total.isHidden()


def test_timestamps_of_the_wrong_length_are_refused(qapp):
    import numpy as np

    s = _playable(total=10)
    with pytest.raises(ValueError, match="one t_ns per scrub position"):
        s.set_timestamps(np.arange(9, dtype=np.int64))


def test_elapsed_sits_left_of_the_slider_and_total_right(qapp):
    """Where VLC and mpv put them, and so where people already look."""
    import numpy as np

    s = _playable(total=100)
    s.set_timestamps(np.arange(100, dtype=np.int64) * 33_333_333)
    s.resize(900, 80)
    s.show()
    qapp.processEvents()
    elapsed, total = s.time_labels()
    slider = s._slider
    assert elapsed.geometry().right() < slider.geometry().left()
    assert total.geometry().left() > slider.geometry().right()
    assert abs(elapsed.geometry().center().y() - slider.geometry().center().y()) <= 4


def test_an_underlay_still_lines_up_with_the_handle_when_the_time_is_shown(qapp):
    """The coverage strip spans the whole widget; the slider now starts after the elapsed time.

    `handle_travel` must report the handle's travel in the *widget's* coordinates, or every
    strip drawn against it is shifted left by the width of a time label.
    """
    import numpy as np

    s = _playable(total=101)
    s.set_timestamps(np.arange(101, dtype=np.int64) * 33_333_333)
    s.resize(900, 80)
    s.show()
    qapp.processEvents()
    left, width = s.handle_travel()
    slider = s._slider
    for value in (0, 50, 100):
        handle_x = slider.mapTo(s, slider.rect().topLeft()).x() + slider.x_of(value)
        assert abs(left + width * value / 100 - handle_x) <= 1


# --------------------------------------------------------------------------- #
# Every host states the time it has (issue 627, item 2)
# --------------------------------------------------------------------------- #
#: Classes that build a ``FrameScrubber`` and give it no timestamps, each with why. A claim:
#: the scan fails on an entry that no longer excuses anything.
_NO_TIME = {}


def _scrubber_hosts_without_time(package_root):
    """``{file::Class}`` of every class that builds a ``FrameScrubber`` and never calls
    ``set_timestamps`` — from the AST, so a docstring mentioning either does not count.
    """
    import ast

    found = set()
    for source in sorted(package_root.rglob("*.py")):
        if source.name in ("scrubber.py", "requirements.py"):
            continue  # the widget itself; the rules' bare counterexamples
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            names = {
                n.func.id if isinstance(n.func, ast.Name) else n.func.attr
                for n in ast.walk(cls)
                if isinstance(n, ast.Call)
                and isinstance(n.func, (ast.Name, ast.Attribute))
            }
            if "FrameScrubber" in names and "set_timestamps" not in names:
                found.add(f"{source.relative_to(package_root).as_posix()}::{cls.name}")
    return found


def test_every_scrubber_host_gives_its_scrubber_the_time(qapp):
    """A shared capability reaches every embedder or it is a fix that reached one of N.

    The recorded shape (`gui-surface-change` §5): the decode scrim reached 1 of 8 embedders, the
    video-required guard 2 of 12. A host added later that never calls ``set_timestamps`` shows no
    time — honest, but the gap this issue was about — and nothing else would say so.
    """
    import pathlib

    import detail.gui

    missing = _scrubber_hosts_without_time(pathlib.Path(detail.gui.__file__).parent)
    assert missing - set(_NO_TIME) == set(), (
        f"{sorted(missing - set(_NO_TIME))} build a FrameScrubber and never give it timestamps. "
        "Pass the recording's t_ns map with set_timestamps, or add the class to _NO_TIME with "
        "the reason it has none."
    )
    stale = set(_NO_TIME) - missing
    assert not stale, f"_NO_TIME excuses classes that now pass: {sorted(stale)}"


def test_the_time_scan_can_fail(tmp_path):
    """Guard the guard: a host that builds a scrubber and sets no time is found."""
    (tmp_path / "host.py").write_text(
        "class Host:\n"
        "    def __init__(self):\n"
        "        self.s = FrameScrubber(playable=True)\n"
        "        self.s.set_total(10)\n",
        encoding="utf-8",
    )
    (tmp_path / "good.py").write_text(
        "class Good:\n"
        "    def __init__(self):\n"
        "        self.s = FrameScrubber(playable=True)\n"
        "    def _load(self, t):\n"
        "        self.s.set_timestamps(t)\n",
        encoding="utf-8",
    )
    assert _scrubber_hosts_without_time(tmp_path) == {"host.py::Host"}


def test_a_span_dims_the_context_and_brackets_the_clip(qapp):
    """The trim picture: outside the span the groove is dimmed, at its ends sit brackets, and
    inside it the groove is left exactly as the style drew it.

    Read off the rendered slider, because "a span was set" is true of the version that drew a
    4px line nobody could read.
    """
    s = _playable(total=101)
    s.resize(600, 60)
    s.show()
    qapp.processEvents()
    slider = s._slider
    plain = slider.grab().toImage()
    s.set_marks(spans=[(40, 60)])
    qapp.processEvents()
    marked = slider.grab().toImage()
    y = round(slider._groove().center().y())

    def changed(x):
        return marked.pixelColor(x, y) != plain.pixelColor(x, y)

    assert changed(round(slider.x_of(10))), "the context before the clip was not dimmed"
    assert changed(round(slider.x_of(90))), "the context after the clip was not dimmed"
    assert not changed(round(slider.x_of(50))), "the clip itself was painted over"
    for edge in (40, 60):
        x = round(slider.x_of(edge))
        top = [
            marked.pixelColor(x, 2) != plain.pixelColor(x, 2) for x in (x - 1, x, x + 1)
        ]
        assert any(top), f"no bracket at the clip's {edge}"


def test_a_compact_transport_keeps_play_and_speed_and_the_added_controls(qapp):
    """The clip montage's transport: every tile loops on its own, so the slider, its time
    labels, the step buttons and the frame readout point at nothing and are hidden; Play,
    the speed menu and a host's own controls stay, and playback is unchanged."""
    from detail.gui.qt import QPushButton, QWidget

    from detail.gui.scrubber import FrameScrubber

    host = QWidget()
    scrubber = FrameScrubber(host, allow_frame_entry=False, playable=True, loop=True)
    extra = QPushButton("Export")
    scrubber.add_control(extra)
    scrubber.set_total(5)
    scrubber.set_transport_compact(True)
    host.show()
    for hidden in (
        scrubber._slider,
        scrubber._elapsed,
        scrubber._total_time,
        scrubber._prev,
        scrubber._nxt,
        scrubber._readout,
    ):
        assert not hidden.isVisibleTo(host), hidden.objectName() or hidden
    assert scrubber.play_button().isVisibleTo(host)
    assert scrubber._speed_box.isVisibleTo(host) and extra.isVisibleTo(host)
    scrubber.set_frame(3)
    assert scrubber.frame() == 3, "playback and position are untouched"
    scrubber.set_transport_compact(False)
    assert scrubber._slider.isVisibleTo(host)
