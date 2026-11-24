"""What the audit player does before, during and after a clip has decoded.

Four faults reported together from one session, and three of them are the same mistake in
different clothes: **something reacting to a frame that is not there yet.**

* The clip **looped**, so it restarted under a decision you were still making.
* Playback **started before the first frame existed**. The decode is off-thread, so the timer
  advanced the counter and ran the scrub bar over an empty pane.
* There was **no loading indication** — and not because one is missing. `decode_overlay` is
  wired, and it is deliberately suppressed *while playing*, so autoplay switched off the only
  thing that would have explained the blank pane. One act caused the symptom and hid its cause.
* The picture **grew for a second or two** after opening a clip, every time you came to the tab
  from another one. `QLabel` takes its size hint from its pixmap, and this pixmap is a *product*
  of the size the label was given — so the hint fed back into the layout that produced it: the
  first frame took the page's hint from 130px to 536px, the pane got more room, the next frame
  was scaled larger. `QSizePolicy.Ignored` is the fix, and it is the right one on its own terms:
  a video pane fills what it is given and should never push back.

The last one only became visible once the content area started sizing pages to their hint
(`gui/scroll.py`), which is what made a pixmap-derived hint load-bearing.
"""

import pathlib
import tempfile

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QPixmap  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from detail.gui.behavior_query import ClipSpec, _ClipPlayer  # noqa: E402
from detail.gui.image_view import ImageView  # noqa: E402


class _Experiment:
    dir = pathlib.Path(tempfile.mkdtemp())


def _player(qapp, **kwargs) -> _ClipPlayer:
    kwargs.setdefault("context_s", 0)
    player = _ClipPlayer(_Experiment(), **kwargs)
    player.resize(600, 400)
    player.show()
    QApplication.processEvents()
    return player


def _clip() -> ClipSpec:
    return ClipSpec("rec", 0, "emb", 0, 2_000_000_000, "rear", score=0.5)


# --- the pane's size must not depend on what is in it ------------------------------- #


def test_the_video_pane_does_not_size_itself_from_its_pixmap(qapp):
    """The growth loop, asserted where it starts.

    On the *page's* hint rather than on the pane's policy: the policy is how it is fixed today,
    the hint not moving is what the user sees. A frame arriving must change nothing about how
    much room anything is asking for.
    """
    pane = ImageView(zoomable=True)
    page = QWidget()
    box = QVBoxLayout(page)
    box.addWidget(pane, 1)
    box.addWidget(QLabel("the controls row under the video"))
    page.resize(900, 600)
    page.show()
    QApplication.processEvents()
    before = page.sizeHint().height()

    frame = QPixmap(1920, 1080)
    for _ in range(5):
        pane.set_frame(frame)
        QApplication.processEvents()

    assert page.sizeHint().height() == before, (
        f"the page's hint went {before} -> {page.sizeHint().height()} because a frame arrived; "
        "the pixmap is a product of the space, so it must not also decide it"
    )


def test_the_pane_still_declares_a_floor(qapp):
    """`Ignored` discards the *hint*, not an explicit minimum — and without a floor a video
    pane in a tight layout collapses to nothing."""
    pane = ImageView(zoomable=True)
    assert pane.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Ignored
    assert pane.minimumHeight() >= 90


# --- nothing runs before there is a frame to run over ------------------------------- #


def test_autoplay_waits_for_a_frame_before_it_starts_counting(qapp):
    """The counter used to advance over an empty pane while the decode was still in flight."""
    player = _player(qapp, autoplay=True)
    player.set_clip(_clip())

    assert (
        not player.is_playing()
    ), "playback started before any frame had been decoded — the scrub bar runs over nothing"
    assert (
        player._start_on_first_frame
    ), "and it is not armed to start when one arrives either"


def test_a_clip_that_cannot_be_decoded_does_not_stay_armed(qapp):
    """Otherwise the *next* frame to arrive from anywhere starts a playback nobody asked for."""
    from detail.gui import frame_source

    player = _player(qapp, autoplay=True)
    player.set_clip(_clip())
    assert (
        player._start_on_first_frame
    ), "the fixture must start armed, or this proves nothing"

    player._display(0, (None, frame_source.UNDECODABLE))

    assert (
        not player._start_on_first_frame
    ), "still armed after a frame that cannot be shown"
    assert not player.is_playing()


# --- and it plays once ---------------------------------------------------------------- #
