"""The strip decodes what is on screen, before what is not.

The class: **an eligibility test asked of a layout that has not run.** ``_rebuild`` creates the
cells and Qt lays them out afterwards, so at the moment the decode pass ran, the row was only
*partially* sized — some cells measured, the ones at the front still reporting ``width() == 0``
at x-values a few pixels apart. Every geometry read was then nonsense, and the "is this cell on
screen" test admitted the whole window.

Measured on a 300-clip behavior with eight cells' worth of viewport: **300 decodes queued at
first paint, in index order**, while the cutoff — the thing being looked at — sits at index 150.
At ~73ms a frame over eight pool threads that is ~1.4s before the visible row shows anything and
~2.7s of churn in total, all of it competing with the clip player for the same global pool. That
is the reported "clip loading is laggy and inconsistent": not one slow decode, but the right
decodes queued behind a hundred and forty wrong ones.

Nothing failed. Every frame arrived, correct, eventually — in an order nobody chose.

Two properties, and both are asserted below because either alone is satisfiable without the
other: the pass is **bounded**, and it is **ordered by what is being looked at**. A test that
only counted would pass on the first 24 of the wrong cells; one that only checked the order
would pass while queueing all 300 of them in a good order.

The second half of this file is that class's sibling: **an eligibility test asked of a widget
that is not on screen yet**, where the answer is not merely wrong but empty. See the section
comment there — it is the whole of #395, and it was reported as "only the clip you click
renders".

**One strip left.** The class was found on the query CALIBRATE's confidence strip and its
ordering moved into ``gui/decode_order.py`` so the clustering strip shared it. The confidence
strip was deleted when CALIBRATE moved to the shared montage (2026-11), whose picture worker has
its own visible-first rule (``tests/gui/test_live_montage.py``); its three tests here went with it.
"""

import pathlib
import tempfile

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QThreadPool  # noqa: E402
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

from detail.gui import decode_order  # noqa: E402
from detail.gui.behavior_query import ClipSpec  # noqa: E402

CLIPS = 300
ON_SCREEN = 8  # what a 1500px-wide strip actually shows at this cell size


@pytest.fixture
def queued(monkeypatch):
    """Every index the strip hands to the thread pool, in order, without running any of them."""
    seen: list[int] = []

    def start(_self, task, *args, **kwargs):
        seen.append(task._index)

    monkeypatch.setattr(QThreadPool, "start", start)
    return seen


def _clips(n: int = CLIPS) -> list:
    return [
        ClipSpec(
            "rec",
            0,
            "emb_01",
            i * 1_000_000,
            i * 1_000_000 + 500_000,
            "rear",
            score=i / n,
        )
        for i in range(n)
    ]


# --- the same two properties, on the other strip ----------------------------------------- #
# The clustering strip fired one task per clip with no ordering at all, while this file's guard
# named `ConfidenceStrip` and could not see it. That is the recorded class *two surfaces
# answering the same question with two mechanisms* — so the ordering moved into
# `gui/decode_order.py` and these assert it over the strip that never had it. A guard naming one
# page cannot see this class, which is why `test_workspace_parity` is written the same way.


def _cluster_strip(qapp):
    from detail.gui.cluster_strip import ClusterClipStrip

    class _Experiment:
        dir = pathlib.Path(tempfile.mkdtemp())

    strip = ClusterClipStrip(_Experiment())
    host = QWidget()
    QVBoxLayout(host).addWidget(strip)
    host.resize(1500, 260)
    host.show()
    QApplication.processEvents()
    strip._host = host  # keep it alive
    return strip


def _cluster_clips(n: int = CLIPS) -> list:
    return [
        ClipSpec(
            "rec", 0, "emb_01", i * 1_000_000, i * 1_000_000 + 500_000, "cluster 3"
        )
        for i in range(n)
    ]


def test_the_cluster_strip_does_not_queue_every_member_at_once(qapp, queued):
    """A cluster hands the strip up to 400 members, and the page draws them into three strips.

    Before this, `show_clips` started one pool task per clip in index order — so a page opened
    ~800 video decodes at once and the pictures on screen were queued behind every picture to
    their left. The bound is the property; the order is the test below.
    """
    strip = _cluster_strip(qapp)
    strip.show_clips(_cluster_clips(), tint=(200, 80, 40))
    QApplication.processEvents()

    assert queued, "nothing was queued at all — the strip decoded nothing"
    assert len(queued) <= decode_order.DECODE_BUDGET, (
        f"{len(queued)} of {CLIPS} members queued at first paint. Decode a screenful; scrolling "
        "asks for the rest."
    )


def test_the_cluster_strip_decodes_the_members_on_screen_first(qapp, queued):
    """Bounded is not enough on its own — the first 24 of the *wrong* cells satisfies that.

    The strip scrolls to the left when its contents change, so what is on screen is the start of
    the row, and those are the indices that must be queued.
    """
    strip = _cluster_strip(qapp)
    strip.show_clips(_cluster_clips(), tint=(200, 80, 40))
    QApplication.processEvents()

    assert max(queued) < CLIPS // 2, (
        f"queued index {max(queued)} of {CLIPS} at first paint, with the row scrolled to the "
        "left — the on-screen filter is being asked of a layout that has not run"
    )


def test_the_cluster_strip_does_not_re_decode_what_it_has(qapp, queued):
    """Scrolling brings in the next screenful and must not re-queue the ones already in flight.

    Without this the scroll handler is a decode amplifier: every pixel of movement re-queues
    everything visible, which is worse than the unordered pass it replaced.
    """
    strip = _cluster_strip(qapp)
    strip.show_clips(_cluster_clips(), tint=(200, 80, 40))
    QApplication.processEvents()
    first = list(queued)

    strip._decode_visible()  # the scroll handler, without moving
    QApplication.processEvents()

    assert queued == first, (
        f"a second pass with nothing moved queued {len(queued) - len(first)} more decodes — "
        "in-flight indices must be skipped"
    )


# --- the same pass, asked before the strip is on screen ----------------------------------- #
# The sibling class: **an eligibility test asked of a widget that is not on screen yet.** The
# cluster strip filtered its cells on `isVisible()`, which is False for every child of a widget
# nobody has shown — so a strip filled *before* it was revealed queued nothing at all, silently,
# and nothing re-checked when it appeared. The interpretation band fills both typicality halves
# and reveals them afterwards, so every cell there stayed on `…` forever unless the row happened
# to be long enough to scroll (#395). Both strips now decode nothing off screen and decode on
# `showEvent`, which is why this is parametrized rather than written against the one that broke.

#: Few enough that the row does not scroll — which is the case the bug needed. A scroll is the
#: one thing that re-ran the pass, so a long row hid this on both strips: the cluster strip's
#: scrollbar eventually moved, and the confidence strip's deferred re-centring moved its bar for
#: it. A short row has neither, and a typicality half-strip is short by construction.
FITS_ON_SCREEN = 6


def _hidden_cluster_strip(qapp):
    strip = _cluster_strip(qapp)
    return strip, lambda: strip.show_clips(
        _cluster_clips(FITS_ON_SCREEN), tint=(200, 80, 40)
    )


@pytest.mark.parametrize("build", [_hidden_cluster_strip], ids=["cluster"])
def test_a_strip_filled_off_screen_decodes_when_it_is_shown(qapp, queued, build):
    """Becoming visible is what asks for the pictures — on a short row, nothing else does.

    Both halves are asserted because either alone is satisfiable without the other: an off-screen
    strip queues nothing (a folded band is not worth a screenful of decodes ordered against a
    viewport it no longer has), and revealing it queues them.
    """
    strip, fill = build(qapp)
    strip.setVisible(False)
    QApplication.processEvents()

    fill()
    QApplication.processEvents()
    assert not queued, (
        f"{len(queued)} decodes queued for a strip nobody can see — they are ordered against a "
        "viewport it does not have"
    )
    assert (
        strip._scroll.horizontalScrollBar().maximum() == 0
    ), "the row scrolls, so a scroll could stand in for the reveal and this proves nothing"

    strip.setVisible(True)
    QApplication.processEvents()
    assert queued, "the strip became visible and never asked for a picture"
