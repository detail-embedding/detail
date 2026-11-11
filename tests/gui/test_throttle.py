"""A drag reports while it moves, at most every 50 ms, and once more on release."""

from detail.gui import throttle
from detail.gui.throttle import Throttle


def _clocked(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(throttle.time, "monotonic", lambda: now[0])
    return now


def test_moves_inside_the_interval_wait_and_release_sends_the_last(monkeypatch):
    now = _clocked(monkeypatch)
    sent: list = []
    live = Throttle(sent.append)
    live.start("a")
    live.offer("b")
    now[0] += 0.01
    live.offer("c")
    now[0] += 0.01
    live.offer("d")
    assert sent == ["b"], "the first move goes at once; the next two are too soon"
    now[0] += 0.05
    live.offer("e")
    assert sent == ["b", "e"]
    live.flush("f")
    assert sent == ["b", "e", "f"], "release sends where the drag ended"


def test_nothing_is_sent_for_a_drag_that_ends_where_it_began(monkeypatch):
    _clocked(monkeypatch)
    sent: list = []
    live = Throttle(sent.append)
    live.start("a")
    live.offer("a")
    live.flush("a")
    assert sent == []


def test_release_does_not_repeat_the_last_live_send(monkeypatch):
    _clocked(monkeypatch)
    sent: list = []
    live = Throttle(sent.append)
    live.start("a")
    live.offer("b")
    live.flush("b")
    assert sent == ["b"]
