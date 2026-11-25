"""The busy / one-shot progress primitives (``detail/gui/busy.py``), and four classes.

``docs/recorded_classes.md`` indexes all four against this file. The write-ups are here because a
guard's module docstring cannot drift from the thing enforcing it.

**1. A wait indicator says the viewer is waiting, not that work is running.** Under a stream those
differ: a decode is always in flight, so a "busy" scrim latches over frames that are arriving
normally. The fix then shipped as an optional ``suppress_when`` predicate **and reached 1 of 8
embedders** — the other seven are all ``playable=True``, so seven surfaces played video under a
permanent "Decoding frame…" clearable only by pausing. The one predicate anybody wrote was
``self._scrubber.is_playing()``: a fact about the argument the function already takes, so it is
derived now and the parameter is gone. A guard over the *mechanism* cannot see this — it passes the
predicate itself. The guard has to ask whether a host gets the behaviour **without opting in**,
which is ``test_every_decode_overlay_embedder_gets_the_suppression``.

**2. A Qt object's C++ side outliving the Python state it needs.** Anything Qt holds by C++ pointer —
a connection, an installed event filter, a queued call — must be outlived by the Python object behind
it, or scoped to the window where something holds it. This file covers the event-filter half
(``test_an_abandoned_overlay_filters_nothing``, ``test_a_running_overlay_is_the_only_one_filtering``);
``tests/gui/test_workers.py`` covers the connection and queued-delivery halves and carries the
signature to recognise it by.

**3. A control that is operable before the state it reads exists.** Every other async-loaded surface
is safe *by accident* — an empty table emits no selection, an empty scatter has no point to click — so
no ``run_busy`` load ever needed to disable anything and the habit was never formed. A slider carries
a meaningful range with no data behind it, and ``BusyOverlay``'s reveal delay was doubling as the
moment input stopped, so the target stayed hit-testable for 150 ms and a press begun there survives
the cover. **Both halves are needed:** a cover cannot stop a focused spin box's ``editingFinished``,
and a *failed* load removes the cover while leaving the state ``None`` for the rest of the session.
``tests/gui/test_cluster_workspace.py`` holds the surface-level half.

**4. An older background result overwriting a newer one.** Overlapping busy calls deliver in the
order their threads *finish*, so whichever read is slowest decides the screen — usually the older
one, which started before the state it should reflect. AUDIT reloaded up to five times while a
behavior was chosen and a cutoff set; a stale "nothing to audit" landing last left Start disabled
after the cutoff had made batches available (13 of 86 full runs of
``test_a_finished_run_freezes_the_cutoff_until_reopened``, and a real wrong screen). Five callers
had their own generation counters and nine had none. ``run_busy`` / ``run_progress`` take
``supersedes=name`` now — the newest call per name on a widget is the only one delivered — and
``requirements.a_reload_supersedes_its_last_call`` makes every reload-shaped method pass one.
Dropping the older result was half of it: the older *work* still ran to its end. A location plot
redraws on every control change, each draw 2–8 s on a 50-recording run, so ticking recordings off
one by one queued a whole draw per tick, all competing for one interpreter, and the picture showed
the last choice seconds after it was made — a hang, to the user. A ``run_progress`` call is now
stopped at its next report by the call that supersedes it
(``test_a_superseded_progress_call_stops_at_its_next_report``); a ``run_busy`` call reports
nothing, so it can only be dropped.
"""

import ast
import pathlib
import time
import types

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import sip  # noqa: E402
from PyQt6.QtWidgets import QWidget  # noqa: E402

import detail.gui.busy as busy  # noqa: E402
from detail.gui.busy import BusyOverlay, run_busy, run_progress  # noqa: E402
from tests.waits import finished, until  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]


def _drain(qapp, worker, timeout_ms=3000):
    """Let the off-thread worker finish, then deliver its queued signals."""
    finished(worker, "the busy call's worker", timeout_ms)
    for _ in range(5):
        qapp.processEvents()


def test_run_busy_delivers_result_and_clears_overlay(qapp):
    host = QWidget()
    host.resize(200, 150)
    got = []
    worker = run_busy(host, lambda: 21 * 2, got.append, label="Loading…")
    assert host.findChildren(BusyOverlay), "overlay should cover the target while busy"
    _drain(qapp, worker)
    assert got == [42]
    # The overlay is *torn down*, not merely hidden. `isVisible()` here was vacuous twice
    # over: `host` is never shown, so it is False for any child regardless, and by this point
    # `clear()` has already `deleteLater()`d the overlay, so the list is empty and `all()` over
    # nothing is True whatever `stop()` does. Assert the teardown, which is the property.
    assert host.findChildren(BusyOverlay) == []
    assert busy._active == set(), "the call was not released from the keep-alive set"


def test_run_busy_surfaces_error_to_on_error(qapp):
    host = QWidget()
    errors = []

    def boom():
        raise ValueError("nope")

    worker = run_busy(host, boom, lambda r: None, on_error=errors.append)
    _drain(qapp, worker)
    assert len(errors) == 1 and isinstance(errors[0], ValueError)


def test_run_busy_drops_result_when_target_torn_down(qapp):
    host = QWidget()
    called = []
    # A slow-enough fn that we can delete the target before it returns.
    worker = run_busy(host, lambda: (time.sleep(0.1) or "x"), called.append)
    sip.delete(host)  # tear the target down synchronously while the worker runs
    _drain(qapp, worker)
    assert called == []  # dropped, and no crash touching the dead widget


def test_run_progress_drives_a_determinate_bar(qapp):
    host = QWidget()
    got = []

    def work(emit):
        emit("exporting", 0.5, "3 / 6")
        return "done"

    host.resize(200, 150)
    worker = run_progress(host, work, got.append, label="Exporting…")
    # Recorded as it happens: the overlay is torn down when the call finishes, so reading the
    # bar afterwards reads a deleted C++ object. Asserting only the result passed with
    # `worker.tick.connect(...)` unwired — the chain that makes the bar *determinate*, which
    # is the name of this test, was never observed at all.
    overlay = host.findChildren(BusyOverlay)[0]
    fractions: list = []
    original = overlay.set_fraction
    overlay.set_fraction = lambda f, text="": (fractions.append(f), original(f, text))[
        1
    ]
    _drain(qapp, worker)
    assert got == ["done"]
    assert fractions == [0.5], f"the bar was never driven: {fractions}"


def test_run_progress_can_say_how_long_is_left(qapp, monkeypatch):
    """An experiment import is minutes, and its fraction is weighed by measured seconds, so the
    overlay can say how long is left (#916). Only when asked: most of these fractions are step
    counts, and an ETA extrapolated from one is a confident wrong number."""
    host = QWidget()
    host.resize(200, 150)
    clock = [100.0]
    # `busy`'s own reference, not the `time` module every other thread is reading.
    monkeypatch.setattr(busy, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))

    def work(emit):
        clock[0] = 110.0  # a quarter of the way after ten seconds
        emit("extracting", 0.25, "1 / 4 files")
        return "done"

    texts: dict[bool, list] = {}
    for eta in (True, False):
        worker = run_progress(host, work, lambda _r: None, eta=eta)
        overlay = host.findChildren(BusyOverlay)[-1]
        seen = texts.setdefault(eta, [])
        original = overlay.set_fraction
        overlay.set_fraction = lambda f, text="", seen=seen, original=original: (
            seen.append(text),
            original(f, text),
        )[1]
        _drain(qapp, worker)
        clock[0] = 100.0

    assert texts[True] == ["extracting — 1 / 4 files · about 30s left"], texts
    assert texts[False] == ["extracting — 1 / 4 files"], texts


def test_no_time_left_is_claimed_before_the_work_has_moved():
    assert busy.with_remaining("reading", 0.0, 5.0) == "reading"
    assert busy.with_remaining("placing", 1.0, 5.0) == "placing"


def _flat_host(color="#3060a0"):
    """A host whose every pixel is one known colour, so "was anything drawn over it" is
    answerable by comparing images rather than by asking a widget whether it feels visible.
    """
    from PyQt6.QtGui import QColor, QPalette

    host = QWidget()
    host.resize(120, 90)
    host.setAutoFillBackground(True)
    palette = host.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(color))
    host.setPalette(palette)
    return host


def _pixels(widget):
    return widget.grab().toImage()


def test_overlay_reveals_after_the_delay_and_hides_on_stop(qapp, monkeypatch):
    monkeypatch.setattr(busy, "SHOW_DELAY_MS", 0)  # reveal on the next event loop turn
    host = _flat_host()
    host.show()
    qapp.processEvents()
    bare = _pixels(host)

    overlay = BusyOverlay(host)
    overlay.start("Loading…")
    qapp.processEvents()  # fire the 0 ms show timer
    assert overlay.isVisible()
    assert _pixels(host) != bare, "past the delay the scrim and label are drawn"

    overlay.stop()
    assert not overlay.isVisible()
    qapp.processEvents()
    assert _pixels(host) == bare, "stopping restores the target's own picture"


def test_the_overlay_covers_its_target_before_it_is_revealed(qapp, monkeypatch):
    """The class: **a wait indicator whose delay also decided when input stopped.**

    ``SHOW_DELAY_MS`` exists so an op that finishes in a blink never flashes — it is about
    appearance. It used to be implemented by leaving the overlay *hidden*, which also left the
    target hit-testable, so for the first 150 ms every ``run_busy`` caller was live over state
    its own load had not produced yet. `_InterpretTab`'s granularity slider raised
    ``AttributeError: 'NoneType' object has no attribute 'minimum'`` through exactly that hole,
    and a mouse press begun inside it survives the cover — Qt grabs the mouse, so the drag runs
    on under the scrim and the release lands seconds into the load.

    So: cover from ``start``, paint from ``_reveal``. ``childAt`` is Qt's own hit test, and it
    skips ``WA_TransparentForMouseEvents``, so it answers the question the bug was about.
    """
    monkeypatch.setattr(busy, "SHOW_DELAY_MS", 10_000)  # never reveals during this test
    host = _flat_host()
    host.show()
    qapp.processEvents()
    centre = host.rect().center()
    bare = _pixels(host)
    assert host.childAt(centre) is None  # nothing over it to begin with

    overlay = BusyOverlay(host)
    overlay.start("Loading…")
    qapp.processEvents()
    assert (
        host.childAt(centre) is overlay
    ), "the target is still reachable during its load"
    assert _pixels(host) == bare, "...but nothing is drawn yet, so nothing flashes"

    overlay.stop()
    qapp.processEvents()
    assert host.childAt(centre) is None, "the cover must lift when the wait ends"


def test_a_decode_overlay_does_not_block_the_view_it_indicates(qapp, monkeypatch):
    """The one opt-out, and the reason it is a choice rather than an oversight.

    A decode indicator sits over a *live* view: a scrubber being dragged is decoding almost
    continuously, so a blocking cover would eat the clicks that drive the next decode. Nothing
    behind it reads state the decode produces, which is what makes it safe — and is not true of
    a ``run_busy`` load.
    """
    monkeypatch.setattr(busy, "SHOW_DELAY_MS", 0)
    from detail.gui.scrubber import FrameScrubber

    host = _flat_host()
    host.show()
    qapp.processEvents()

    scrubber = FrameScrubber(playable=False)
    overlay = busy.decode_overlay(scrubber, host)
    scrubber.busy_changed.emit(True)
    qapp.processEvents()
    assert overlay.isVisible(), "the indicator is shown — it is not suppressed here"
    assert host.childAt(host.rect().center()) is None, "and does not intercept the view"


def test_overlay_set_fraction_switches_to_determinate(qapp):
    host = QWidget()
    overlay = BusyOverlay(host)
    overlay.start("Working…")
    overlay.set_fraction(0.5, "half")
    assert overlay._bar.maximum() == 1000 and overlay._bar.value() == 500


# --------------------------------------------------------------------------- #
# a wait indicator over a stream
# --------------------------------------------------------------------------- #
# The class: **a wait indicator whose condition is "work is in flight" rather than "the user
# is waiting with nothing to look at".** `FrameScrubber.busy_changed` says a decode is running
# and stays True across a re-dispatch burst — right for a scrub, where the picture is frozen
# and the user is waiting. Under autoplay a decode is *always* running: `_advance` requests one
# every timer tick, `_on_decoded` dispatches straight into the next queued frame, and
# `busy_changed(False)` arrives only when the queue drains. A player that cannot quite keep up
# never drains it — backgrounding the app guarantees this, since the timer catches up in a
# burst on return — so the 150ms reveal fires once and nothing ever hides it again. The result
# is a scrim reading "Decoding frame…" over frames that are arriving 25 times a second, and
# pausing (which drains the queue) is the only way to clear it.
#
# Nothing fails. The frames are decoded, the player plays, and the only symptom is a permanent
# label claiming the opposite of what is happening.
def _scrubber_and_overlay(qapp, monkeypatch, *, playing):
    """A real playable scrubber, actually playing or not — never a stand-in predicate.

    The previous version passed ``suppress_when=lambda: playing`` and built the scrubber
    ``playable=False``. That tests the mechanism and can never ask the question that mattered:
    *does a host get this behaviour without opting in?* Seven of eight embedders did not, and
    both tests below passed throughout. Drive ``is_playing()`` for real, which is the only
    thing ``decode_overlay`` now consults.
    """
    monkeypatch.setattr(busy, "SHOW_DELAY_MS", 0)
    from detail.gui.scrubber import FrameScrubber

    host = QWidget()
    host.resize(120, 90)
    host.show()
    scrubber = FrameScrubber(playable=True)
    # Playback refuses to start on a scrubber with nothing to play, so give it both the
    # length and a rendered frame — what a decoded clip does (see the `_ClipPlayer` test).
    scrubber.set_total(10)
    scrubber._has_rendered = True
    overlay = busy.decode_overlay(scrubber, host)
    if playing:
        scrubber.play()
        assert scrubber.is_playing(), "the fixture failed to start playback"
    else:
        assert not scrubber.is_playing()
    return scrubber, overlay


def test_no_decode_scrim_over_a_playing_clip(qapp, monkeypatch):
    scrubber, overlay = _scrubber_and_overlay(qapp, monkeypatch, playing=True)
    scrubber.busy_changed.emit(True)
    qapp.processEvents()
    assert not overlay.isVisible()


def test_a_decode_the_viewer_is_waiting_on_still_shows(qapp, monkeypatch):
    """The negative case. Without it the suppression passes vacuously — an overlay that never
    appears satisfies "does not appear while playing" perfectly."""
    scrubber, overlay = _scrubber_and_overlay(qapp, monkeypatch, playing=False)
    scrubber.busy_changed.emit(True)
    qapp.processEvents()
    assert overlay.isVisible()
    scrubber.busy_changed.emit(False)
    assert not overlay.isVisible()


def test_every_decode_overlay_embedder_gets_the_suppression(qapp, monkeypatch):
    """The class, and why the two tests above could not see it.

    Suppression arrived as ``suppress_when``, an optional parameter defaulting to "never
    suppress". **One of eight embedders passed it.** The other seven all build
    ``FrameScrubber(playable=True)`` — the detection and tracking frame viewers, both stream
    previews, scene geometry, the query example player, query-eval's Explore clips — so seven
    surfaces played video under a permanent "Decoding frame…" scrim, clearable only by pausing.

    So the guard is not "the mechanism works" but "no embedder can be without it": the only
    input is the scrubber, which every caller passes positionally because it is what the
    overlay is *for*. A predicate parameter coming back is the regression.
    """
    import inspect

    params = inspect.signature(busy.decode_overlay).parameters
    assert list(params) == ["scrubber", "target", "label"], (
        "decode_overlay grew a parameter. If it is a host-supplied predicate for whether the "
        "scrim applies, that is the bug this test exists for — derive it from `scrubber`."
    )
    # And every call site passes the scrubber, so every one is covered by construction.
    call_sites = []
    for path in sorted((REPO / "detail" / "gui").rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None))
                == "decode_overlay"
            ):
                call_sites.append((path.relative_to(REPO).as_posix(), node))
    assert len(call_sites) >= 8, (
        f"only found {len(call_sites)} decode_overlay call sites — the sweep is broken, and "
        "an empty sweep would pass every assertion below it"
    )
    for rel, node in call_sites:
        assert node.args, f"{rel}: decode_overlay called with no scrubber"
        extra = {k.arg for k in node.keywords} - {"label"}
        assert (
            not extra
        ), f"{rel}: decode_overlay passed {sorted(extra)} — see this test"


def test_the_player_suppresses_its_own_overlay_while_playing(
    qapp, tmp_path, monkeypatch
):
    """Through the real construction, not a hand-wired pair: the predicate has to reach the
    player's own play state, and the overlay is built before the button it reads."""
    monkeypatch.setattr(busy, "SHOW_DELAY_MS", 0)
    from detail.api.experiment import Experiment
    from detail.gui.behavior_query import _ClipPlayer

    player = _ClipPlayer(Experiment.create("p", tmp_path), context_s=0)
    player.resize(200, 150)
    player.show()
    # Length and "a frame is up" now live on the scrubber, which owns playback — setting
    # `_len` alone leaves its total at 0, and a player with nothing to play snaps Play straight
    # back. Give it both, which is what a decoded clip does.
    player._len = 10
    player._scrubber.set_total(10)
    player._scrubber._has_rendered = True
    player.play()
    assert player.is_playing()
    player._scrubber.busy_changed.emit(True)
    qapp.processEvents()
    assert not player._decode_busy.isVisible()

    player.pause()
    player._scrubber.busy_changed.emit(True)
    qapp.processEvents()
    assert player._decode_busy.isVisible()


def test_an_abandoned_overlay_filters_nothing(qapp):
    """The class again: **C++ state outliving the Python object it needs.**

    A `BusyOverlay` is parented to its target, so the C++ object and any filter installed on it
    outlive the last Python reference to the wrapper. Qt then delivers a resize into a wrapper it
    materializes fresh — one that never ran `__init__` — and `self._target` does not exist:

        AttributeError: 'BusyOverlay' object has no attribute '_target'

    It escaped into the excepthook, where it was swallowed, and surfaced only once the GUI
    teardown started draining the thread pool and delivering inside the test that caused it.

    The filter is scoped to `start`/`stop` now, so an overlay nobody holds is filtering nothing.

    **Asserted on the filter being installed, not on an exception.** The first version only
    resized after `del` and relied on the excepthook guard catching the `AttributeError` — and
    stayed green against the full pre-fix `BusyOverlay`, because sip keeps a parented wrapper
    alive, so no fresh one is materialized in a test. What the fix changed is *whether the filter
    is installed after `stop`*, and a sent event reaches installed filters synchronously, shown or
    not.
    """
    import gc

    from PyQt6.QtCore import QEvent, QSize
    from PyQt6.QtGui import QResizeEvent
    from PyQt6.QtWidgets import QApplication, QWidget

    from detail.gui.busy import BusyOverlay

    class _Recording(BusyOverlay):
        seen: list = []

        def eventFilter(self, obj, event):  # noqa: N802 - Qt virtual
            if obj is getattr(self, "_target", None):
                _Recording.seen.append(event.type())
            return super().eventFilter(obj, event)

    def _resize_sent():
        _Recording.seen.clear()
        QApplication.sendEvent(target, QResizeEvent(QSize(310, 160), QSize(300, 150)))
        return QEvent.Type.Resize in _Recording.seen

    target = QWidget()
    target.resize(200, 100)

    overlay = _Recording(target)
    assert not _resize_sent(), "an overlay that has not started filters nothing"
    overlay.start("Loading…")
    assert overlay._target is target
    assert (
        _resize_sent()
    ), "a running overlay must see its target resize (or this is vacuous)"

    overlay.stop()
    assert not _resize_sent(), (
        "a stopped overlay is still installed as its target's event filter, so it outlives "
        "whatever holds it and Qt delivers into a wrapper with no instance state"
    )
    del overlay
    gc.collect()

    # A resize with no Python reference left: the filter must be gone, so nothing is delivered
    # into a wrapper without instance state.
    target.resize(300, 150)
    qapp.processEvents()


def test_a_running_overlay_is_the_only_one_filtering(qapp):
    """Scoping the filter must not cost the behaviour it exists for.

    Asserted through the filter itself rather than through a geometry change: an offscreen widget
    that is never shown does not deliver the resize, so a geometry assertion here passes or fails
    for reasons that have nothing to do with the filter — it reads the same before and after this
    change.
    """
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QWidget

    from detail.gui.busy import BusyOverlay

    target = QWidget()
    target.resize(200, 100)
    overlay = BusyOverlay(target)

    overlay.start("Loading…")
    target.resize(320, 180)
    overlay.eventFilter(target, QEvent(QEvent.Type.Resize))
    assert overlay.geometry().width() == 320, "a running overlay follows its target"

    overlay.stop()
    target.resize(200, 100)
    overlay.eventFilter(target, QEvent(QEvent.Type.Resize))
    assert (
        overlay.geometry().width() == 200
    ), "the filter still works if called directly"


def _overlapping(qapp, start, key):
    """Start a slow call, then a fast one, release the slow one; return what was delivered."""
    import threading

    from PyQt6.QtTest import QTest

    host = QWidget()
    got, failed = [], []
    gate = threading.Event()
    older = start(host, lambda: (gate.wait(5), "older")[1], got.append, failed, key)
    newer = start(host, lambda: "newer", got.append, failed, key)
    finished(newer, "the newer read")
    for _ in range(20):
        qapp.processEvents()
        QTest.qWait(5)
    gate.set()  # the older read finishes last
    finished(older, "the older read")
    for _ in range(20):
        qapp.processEvents()
        QTest.qWait(5)
    assert busy._active == set(), "a superseded call was not released"
    assert host.findChildren(BusyOverlay) == [], "a superseded call left its overlay up"
    return got, failed


def _busy(host, fn, done, failed, key):
    return run_busy(host, fn, done, on_error=failed.append, supersedes=key)


def _progress(host, fn, done, failed, key):
    return run_progress(
        host, lambda emit: fn(), done, on_error=failed.append, supersedes=key
    )


@pytest.mark.parametrize("start", [_busy, _progress], ids=["run_busy", "run_progress"])
def test_the_newest_call_wins_whatever_order_the_threads_finish_in(qapp, start):
    got, failed = _overlapping(qapp, start, "audit")
    assert got == [
        "newer"
    ], f"delivered {got}: the older read landed last and overwrote it"
    assert failed == []


def test_without_a_name_both_results_land_in_finishing_order(qapp):
    """The old contract stands for callers that do not opt in — and shows what the name stops."""
    got, _failed = _overlapping(qapp, _busy, None)
    assert got == ["newer", "older"]


def test_a_superseded_error_is_dropped_too(qapp):
    import threading

    from PyQt6.QtTest import QTest

    host = QWidget()
    got, failed = [], []
    gate = threading.Event()

    def boom():
        gate.wait(5)
        raise RuntimeError("stale")

    older = run_busy(host, boom, got.append, on_error=failed.append, supersedes="x")
    run_busy(host, lambda: "fresh", got.append, on_error=failed.append, supersedes="x")
    gate.set()
    finished(older, "the stale read")
    for _ in range(40):
        qapp.processEvents()
        QTest.qWait(5)
    assert got == ["fresh"]
    assert failed == [], "an error from a read nobody is waiting for any more was shown"


def test_names_are_per_widget(qapp):
    """Two widgets loading the same kind of thing do not supersede each other."""
    from PyQt6.QtTest import QTest

    first, second = QWidget(), QWidget()
    got = []
    a = run_busy(first, lambda: "first", got.append, supersedes="audit")
    b = run_busy(second, lambda: "second", got.append, supersedes="audit")
    finished(a, "the first widget's read")
    finished(b, "the second widget's read")
    for _ in range(20):
        qapp.processEvents()
        QTest.qWait(5)
    assert sorted(got) == ["first", "second"]


def test_a_superseded_progress_call_stops_at_its_next_report(qapp):
    """The second half of class 4: the newest call wins *and* the ones it superseded stop.

    Dropped at delivery but left running, every superseded call ran to its end first: a run of
    quick changes cost one whole job per change, and the newest waited behind them all.
    """
    host = QWidget()
    reached, got, failed = [], [], []
    older = run_progress(
        host, _long_work(reached), got.append, on_error=failed.append, supersedes="plot"
    )
    until(qapp, lambda: reached, "the older call reaching its first report")
    newer = run_progress(
        host,
        lambda emit: "newer",
        got.append,
        on_error=failed.append,
        supersedes="plot",
    )
    finished(older, "the superseded call")
    _drain(qapp, newer)
    assert len(reached) < 2000, "the superseded call ran to its end"
    assert got == ["newer"] and failed == [], (got, failed)


def test_a_call_under_another_name_or_widget_is_not_stopped(qapp):
    host, other = QWidget(), QWidget()
    reached, got = [], []
    older = run_progress(host, _long_work(reached, 50), got.append, supersedes="plot")
    until(qapp, lambda: reached, "the first call reaching its first report")
    run_progress(host, lambda emit: "named", got.append, supersedes="table")
    run_progress(other, lambda emit: "elsewhere", got.append, supersedes="plot")
    _drain(qapp, older)
    until(qapp, lambda: len(got) == 3, "all three results")
    assert len(reached) == 50 and sorted(got) == ["elsewhere", "finished", "named"]


# --- cancel (reported from the app: Add frames could not be stopped, and ran twice) --------- #
def _long_work(reached, stop_after=2000):
    def work(emit):
        for i in range(stop_after):
            reached.append(i)
            emit("copying frames", i / stop_after, f"{i}/{stop_after}")
            time.sleep(0.002)
        return "finished"

    return work


def test_cancel_stops_the_work_at_its_next_report(qapp):
    host = QWidget()
    host.resize(200, 150)
    reached, done, errors, cancelled = [], [], [], []
    worker = run_progress(
        host,
        _long_work(reached),
        done.append,
        on_error=errors.append,
        on_cancel=lambda: cancelled.append(True),
    )
    overlay = host.findChildren(BusyOverlay)[0]
    assert overlay._cancel is not None, "a cancellable run offers Cancel"
    until(qapp, lambda: reached, "the work reaching its first report")
    overlay._cancel.click()
    assert overlay._label.text() == "Cancelling…"
    _drain(qapp, worker)
    assert cancelled == [True] and done == [] and errors == [], (done, errors)
    assert len(reached) < 2000, "the work ran to the end"


def test_a_run_without_on_cancel_offers_none(qapp):
    host = QWidget()
    worker = run_progress(host, lambda emit: "x", lambda r: None)
    from PyQt6.QtWidgets import QPushButton

    overlay = host.findChildren(BusyOverlay)[0]
    assert (
        overlay.findChildren(QPushButton) == []
    ), "an overlay that offers no Cancel holds no button"
    _drain(qapp, worker)


def test_closing_the_window_cancels_its_run(qapp):
    """Closing a dialog by Close or Esc emits `finished`, not a close event; either way the
    work stops, so opening the window again never runs a second copy beside the first.
    """
    from PyQt6.QtWidgets import QDialog, QVBoxLayout

    dialog = QDialog()
    body = QWidget()
    QVBoxLayout(dialog).addWidget(body)
    reached, done, cancelled = [], [], []
    worker = run_progress(
        body, _long_work(reached), done.append, on_cancel=lambda: cancelled.append(True)
    )
    until(qapp, lambda: reached, "the work reaching its first report")
    dialog.reject()
    finished(worker, "the run its window closed on")
    assert worker.isFinished() and len(reached) < 2000 and done == []


def test_stop_still_keeps_a_partial_result(qapp):
    """`request_stop` is the other thing (finish early, keep what you have, polled by the work)
    and must not turn into a cancel: comparisons and tool runs rely on it."""
    from detail.gui.workers import StageProgressWorker

    def work(emit):
        emit("a", 0.5, "")
        return "partial"

    worker = StageProgressWorker(work, receiver=None)
    worker.request_stop()
    got = []
    worker.done.connect(got.append)
    worker.start()
    _drain(qapp, worker)
    assert got == ["partial"]
