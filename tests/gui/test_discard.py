"""A widget a rebuild removes leaves the page at once, not when the event loop next turns.

**The class.** ``deleteLater`` schedules a deletion; until deferred deletes run, the widget is
still a visible child of its parent — in its layout, returned by ``findChildren``, drawn. A bare
``processEvents`` does not run them. So a page that rebuilt part of itself by ``deleteLater``-ing
the old widgets held *both* sets for a while, and anything that looked the new one up — a test, an
accessibility walk, a ``findChildren`` in the page's own code — could find the old one first.

**How it showed.** ``RecordingDetailPage`` rebuilds its scene-geometry block after a video part is
added. ``test_recording_detail_preview_follows_an_added_video_part`` found "Edit scene geometry…"
by text and got the old, disabled button: 9 of 86 full runs, whenever the frame decode finished
before the test's first pump (load makes that likely). Forcing that order failed it 3 of 3.

**The rule.** Remove a widget with ``lifetime.discard`` — hidden, unparented (which also takes it
out of its layout), then ``deleteLater``. Seventeen rebuild sites across eight files did their own
version, and only two of them unparented first. ``tests/gui/conftest.py::drain_until`` also
delivers deferred deletes on each pump, for anything that still ``deleteLater``-s directly.
"""

from detail.gui.qt import (
    QEvent,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    is_deleted,
)

from detail.api.experiment import Experiment
from detail.gui.lifetime import discard


def test_a_discarded_widget_is_off_the_page_before_the_loop_turns(qapp):
    page = QWidget()
    layout = QVBoxLayout(page)
    old = QLabel("old")
    layout.addWidget(old)
    page.show()
    qapp.processEvents()

    discard(old)

    assert old not in page.findChildren(QLabel)
    assert layout.count() == 0
    assert not old.isVisible()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert is_deleted(old)


def test_discarding_nothing_or_a_deleted_widget_is_a_no_op(qapp):
    gone = QLabel()
    gone.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    discard(gone)
    discard(None)


def test_a_rebuilt_scene_geometry_block_holds_one_editor_button(qapp, tmp_path):
    """The reported shape, with no event-loop turn between rebuilds to hide it."""
    from detail.gui import pages

    proj = Experiment.create("p", tmp_path)
    page = pages.RecordingDetailPage(proj.create_recording("s1"), experiment=proj)
    page.show()
    qapp.processEvents()

    page._rebuild_scene_geometry()
    page._rebuild_scene_geometry()

    editors = [
        b for b in page.findChildren(QPushButton) if b.text() == "Edit scene geometry…"
    ]
    assert len(editors) == 1, f"{len(editors)} editor buttons after two rebuilds"
