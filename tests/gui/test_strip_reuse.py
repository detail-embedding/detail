"""The clip strip re-points its cells; it does not rebuild them.

The class: **rebuilding widgets to show pictures you already have.** ``_rebuild`` deleted every
widget in the row and constructed fresh ones, and the row is up to ``2 * _WINDOW`` cells wide — so
a behavior switch and *every cutoff nudge* paid for ~300 ``_Cell`` constructions and the layout
pass after them. The frames themselves were already cached, so none of that work produced a
picture that was not already in hand.

Measured on a 300-cell window: a cutoff nudge went from 31–37ms to 15–20ms, and the first paint
from 60ms to 36ms.

Two things this file pins, and the second is the one that nearly shipped as a regression:

**The cells are reused.** Asserted by identity — the same Python objects appear after a nudge —
because "it is faster" is a wall-clock claim that flakes on a busy machine, and the property that
makes it faster is that nothing was rebuilt.

**A pooled cell is parented when it is made.** A widget constructed with no parent is a *top-level
window*, and macOS gives each one native backing, so building the pool that way cost far
more than the same constructions straight into a layout. The first version of this
optimisation was much slower than the thing it replaced, and every behavioural test
passed on it — the widgets were correct, there were simply hundreds of windows.
"""

import pathlib
import tempfile

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QThreadPool  # noqa: E402
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

import detail.gui.clip_grid as clip_grid  # noqa: E402
from detail.gui.behavior_query import ClipSpec  # noqa: E402

N = 400


@pytest.fixture
def strip(qapp, monkeypatch):
    """A laid-out strip over a window big enough that a rebuild is worth avoiding."""
    monkeypatch.setattr(QThreadPool, "start", lambda *a, **k: None)  # no real decoding

    class _Experiment:
        dir = pathlib.Path(tempfile.mkdtemp())

    widget = clip_grid.ConfidenceStrip(_Experiment())
    host = QWidget()
    QVBoxLayout(host).addWidget(widget)
    host.resize(1500, 260)
    host.show()
    QApplication.processEvents()
    widget._host = host  # keep it alive
    widget.show_clips(
        [
            ClipSpec(
                "rec",
                0,
                "emb",
                i * 1_000_000,
                i * 1_000_000 + 500_000,
                "rear",
                score=i / N,
            )
            for i in range(N)
        ],
        0.5,
    )
    QApplication.processEvents()
    return widget


def test_a_cutoff_nudge_reuses_the_cells_rather_than_rebuilding_them(strip):
    """By identity. The frames were already cached, so a rebuild bought nothing but the bill."""
    before = {id(cell) for cell in strip._cells.values()}
    assert len(before) > 50, "the fixture's window is too small to be worth measuring"

    strip.set_cutoff(0.5 + 3.0 / N)
    QApplication.processEvents()

    after = {id(cell) for cell in strip._cells.values()}
    shared = before & after
    assert len(shared) > len(after) * 0.8, (
        f"only {len(shared)} of {len(after)} cells survived a one-clip cutoff nudge — the row "
        "is being rebuilt again"
    )


def test_the_pool_only_ever_grows_to_the_window(strip):
    """Never shrinks, never leaks: a hidden cell costs nothing against building it again, and a
    pool that grew per nudge would be the same bug wearing a cache."""
    first = len(strip._pool)
    for k in range(5):
        strip.set_cutoff(0.5 + (k + 1) / N)
        QApplication.processEvents()
    assert (
        len(strip._pool) == first
    ), f"the pool grew from {first} to {len(strip._pool)}"
    assert first <= 2 * clip_grid._WINDOW + 1


def test_growing_the_pool_creates_no_top_level_windows(strip):
    """A parentless widget **is** a window, and macOS gives each one native backing.

    Built that way, the pool was a regression on the very thing this optimisation exists
    for, and every behavioural test was perfectly happy with it.

    Counted at the application, not asserted on the cells. By the time `_lay_out_row` has run
    every cell has a parent either way, so a test that reads the finished state cannot tell the
    two apart — the first version of this test did exactly that and passed against the
    parentless build. It has to catch them while they are being made.
    """
    before = len(QApplication.topLevelWidgets())

    strip._ensure_cells(len(strip._pool) + 40)

    assert len(QApplication.topLevelWidgets()) == before, (
        "growing the pool created top-level windows — the cells are being constructed without "
        "a parent and reparented afterwards, which pays for the window anyway"
    )


def test_the_divider_lands_where_it_is_measured(strip):
    """The reuse's own trap, and why it was reverted once before.

    Centring reads the divider's laid-out position. With the cells reused a rebuild can leave the
    row the same width, so Qt has no reason to have run the layout — and the row is *inside a
    ``widgetResizable`` scroll area*, which sizes it, so activating only the row's own layout
    measures it at its old width. Traced on a cut: 28760 read against a row that settled at
    28800, leaving the scrollbar 40px shy with nothing scheduled to correct it.
    """
    bar = strip._scroll.horizontalScrollBar()
    strip.set_cutoff(0.75)
    QApplication.processEvents()
    strip.back_to_cutoff()
    for _ in range(10):
        QApplication.processEvents()

    handle = strip._handle
    assert handle is not None
    middle = handle.x() + handle.width() // 2
    want = max(0, min(bar.maximum(), middle - strip._scroll.viewport().width() // 2))
    assert bar.value() == want, f"divider off centre: scroll {bar.value()}, want {want}"


# --- the view belongs to the cutoff until the reader takes it elsewhere ---------------- #


def _centred(strip) -> tuple[int, int]:
    bar = strip._scroll.horizontalScrollBar()
    handle = strip._handle
    middle = handle.x() + handle.width() // 2
    want = max(0, min(bar.maximum(), middle - strip._scroll.viewport().width() // 2))
    return bar.value(), want


def test_results_that_arrive_before_the_real_width_still_end_up_centred(
    qapp, monkeypatch
):
    """Opening a run **is** a resize, and that is the case that was broken.

    The search results land while the page is still being laid out, so the strip centres against
    whatever width it has at that moment — and nothing re-centred when the real one arrived.
    Measured on a 2000-clip ranking: the divider ended up **497px off**, about two and a half
    cells, which is exactly far enough to be looking at clips that have not decoded yet with the
    cutoff off screen. Hence "I have to press *centre on cutoff* myself".

    The order is the whole test: shown small, clips, *then* the real size.
    """
    monkeypatch.setattr(QThreadPool, "start", lambda *a, **k: None)

    class _Experiment:
        dir = pathlib.Path(tempfile.mkdtemp())

    strip = clip_grid.ConfidenceStrip(_Experiment())
    host = QWidget()
    QVBoxLayout(host).addWidget(strip)
    host.resize(400, 300)  # the page is still being built
    host.show()
    QApplication.processEvents()

    big = 2000
    strip.show_clips(
        [
            ClipSpec(
                "rec",
                0,
                "emb",
                i * 1_000_000,
                i * 1_000_000 + 500_000,
                "rear",
                score=i / big,
            )
            for i in range(big)
        ],
        0.5,
    )
    QApplication.processEvents()
    host.resize(1500, 300)  # ...and now the page lays out at its real width
    for _ in range(10):
        QApplication.processEvents()

    value, want = _centred(strip)
    assert (
        value == want
    ), f"the divider is {want - value}px off centre after the page sized itself"


def test_a_reader_who_scrolled_away_is_not_dragged_back_by_a_resize(strip):
    """The negative case, and the reason this is guarded rather than unconditional: following
    the cutoff through a resize must not mean yanking the view off whatever you went to look at.
    """
    bar = strip._scroll.horizontalScrollBar()
    assert bar.maximum() > 0, "the fixture's row does not scroll — this proves nothing"
    bar.setValue(max(bar.value() - 2000, 0))
    QApplication.processEvents()
    away = bar.value()

    strip._host.resize(1300, 260)
    for _ in range(10):
        QApplication.processEvents()

    assert (
        abs(bar.value() - away) < 200
    ), f"a resize moved the view from {away} to {bar.value()} — the reader had scrolled there"


def test_moving_the_cutoff_re_adopts_the_view(strip):
    """The window is a function of the cutoff, so moving it hands the view back to the line —
    otherwise one scroll would opt out of centring for the rest of the session."""
    bar = strip._scroll.horizontalScrollBar()
    bar.setValue(max(bar.value() - 2000, 0))
    QApplication.processEvents()

    strip.set_cutoff(0.75)
    for _ in range(10):
        QApplication.processEvents()

    value, want = _centred(strip)
    assert value == want, f"the divider is {want - value}px off after the cutoff moved"
