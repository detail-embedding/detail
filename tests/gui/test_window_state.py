"""The window opens where it was left, and never off the screen.

Two things this file pins, and the first one is embarrassing rather than subtle: **the docs have
claimed for a long time that window size and the rail split are persisted, and nothing in
``detail/gui/`` called ``saveGeometry``, ``restoreGeometry``, ``saveState`` or ``restoreState``.**
A rule that is written down, believed, and not implemented is worse than one that was never
written, because nobody goes looking. The scan below is the half that keeps the claim honest.

The second is the reason ``restoreGeometry`` is used rather than a remembered width and height.
Qt's own documentation is explicit that it *"checks if the restored geometry is outside the
available screen geometry, and modifies it as appropriate"* — which is exactly the clamping
``_opening_size`` does by hand, done properly, and it is what matters when a laptop is opened
again without the external monitor it was last docked to. A remembered 2560x1400 restored onto a
1470x833 screen is a window whose title bar cannot be reached.

What is *not* persisted, deliberately: per-page state. ``MainWindow._discard_content`` rebuilds
the page on every navigation, so a splitter dragged inside one is gone as soon as you click
elsewhere — persisting it would mean holding state for pages that no longer exist. The rail split
survives because it belongs to the shell rather than to a page, and the result bands survive
through their own keys (``tests/gui/test_collapse.py``).
"""

import ast
import pathlib

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QByteArray, QSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

_APP = pathlib.Path(__file__).resolve().parents[2] / "detail" / "gui" / "app.py"


@pytest.fixture(autouse=True)
def _clean_settings():
    """These keys are machine-global, so a test that writes one would otherwise decide what the
    next test — and the developer's own next launch — opens with."""

    def clear():
        settings = QSettings()
        settings.remove("window")
        settings.remove("panes/navigation")

    clear()
    yield
    clear()


def _window(tmp_path, monkeypatch):
    from detail.gui import app as gui_app
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    return gui_app.MainWindow()


def test_the_shell_actually_calls_qts_geometry_api():
    """The claim, checked against the source rather than the docs.

    A behavioural test alone would not have caught the original state: there was no persistence
    at all, and every *other* test passed, because a window that never restores anything looks
    exactly like one whose saved state happens to match the default.
    """
    source = _APP.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    # The window's own geometry; the rail's split is a `GripSplitter`, which remembers itself
    # once `remember` is called (`tests/gui/test_grip_splitter.py`).
    for name in ("saveGeometry", "restoreGeometry", "remember"):
        assert name in called, f"the shell never calls {name}; nothing is persisted"


def _a_size_this_screen_can_hold():
    """A test size inside the *running* screen, not a number picked on a desktop.

    ``restoreGeometry`` clamps to the available screen — which is the point of using it — and
    the offscreen platform CI runs on reports an 800x800 virtual screen. Asserting a remembered
    1100px width there fails on Qt doing exactly the right thing.
    """
    available = QApplication.primaryScreen().availableGeometry()
    return min(700, available.width() - 60), min(600, available.height() - 60)


def test_the_window_remembers_its_size(qapp, tmp_path, monkeypatch):
    width, height = _a_size_this_screen_can_hold()
    window = _window(tmp_path, monkeypatch)
    window.show()
    window.resize(width, height)
    QApplication.processEvents()
    window._save_layout()
    size = window.size()

    reopened = _window(tmp_path, monkeypatch)
    # Shown before the assertion: an un-shown window has not yet run layout, so
    # `restoreGeometry` reports whatever it clamped the *restored* geometry to against
    # the available screen — not the size Qt actually enforces once the window's real
    # minimum (computed from live font metrics) is applied. That gap is invisible on a
    # screen comfortably larger than the app's minimum size, and exactly this test's
    # shape once the offscreen QPA platform's small virtual screen and font metrics push
    # the minimum width close to or past it.
    reopened.show()
    QApplication.processEvents()
    assert reopened.size() == size, "the window opened at a size it was not left at"


def test_the_rail_divider_stays_where_it_was_dragged(qapp, tmp_path, monkeypatch):
    """The other half of the claim, and the one a user notices: a rail widened to read long
    recording names narrowed itself again at every launch."""
    width, height = _a_size_this_screen_can_hold()
    window = _window(tmp_path, monkeypatch)
    window.show()
    window.resize(width, height)
    # Quick tools, because Home renders full-width with no rail at all — the splitter's first
    # pane is 0px there and "widen it" has nothing to widen.
    window._switch_mode("tools")
    QApplication.processEvents()
    default = window._split.sizes()[0]
    # Dragged by its grip, the way a user moves it — narrower, since it opens at its widest.
    window._split.drag_handle(1, default - 60)
    QApplication.processEvents()
    dragged = window._split.sizes()[0]
    window.close()
    assert dragged < default - 30, "the fixture never actually moved the rail"

    reopened = _window(tmp_path, monkeypatch)
    reopened.show()
    reopened.resize(width, height)
    reopened._switch_mode("tools")
    QApplication.processEvents()
    restored = reopened._split.sizes()[0]
    assert (
        abs(restored - dragged) <= 2
    ), f"the rail reopened at {restored}px — it was left at {dragged} and defaults to {default}"
    reopened.close()


def test_a_first_launch_opens_at_a_size_the_screen_can_hold(
    qapp, tmp_path, monkeypatch
):
    """With nothing remembered, the default applies — and it is bounded, because `resize` does
    not fail on a size the display cannot hold, it just puts the title bar out of reach.
    """
    window = _window(tmp_path, monkeypatch)
    available = QApplication.primaryScreen().availableGeometry()
    assert window.width() <= available.width()
    assert window.height() <= available.height()


def test_a_corrupt_saved_geometry_falls_back_rather_than_being_trusted(
    qapp, tmp_path, monkeypatch
):
    """``restoreGeometry`` returns False on a blob it cannot read, and OBS's lesson is to check
    it: unchecked, a stale or truncated value silently leaves the window at whatever size it
    happened to have, which on a first paint is 640x480."""
    QSettings().setValue("window/geometry", QByteArray(b"not a geometry"))

    window = _window(tmp_path, monkeypatch)

    available = QApplication.primaryScreen().availableGeometry()
    assert window.width() <= available.width()
    assert window.height() > 480, (
        f"fell back to {window.width()}x{window.height()} — a rejected blob must give the "
        "default, not whatever Qt left behind"
    )


def test_the_layout_is_saved_on_the_way_out(qapp, tmp_path, monkeypatch):
    """Saved in ``closeEvent``, before the in-flight-runs prompt rather than after it: two of
    the three ways out of that prompt quit, and a layout that only persisted on the quietest of
    them is one a user would find had reverted at random."""
    from PyQt6.QtGui import QCloseEvent

    window = _window(tmp_path, monkeypatch)
    window.show()
    window.resize(1150, 720)
    QApplication.processEvents()
    window.closeEvent(QCloseEvent())

    assert QSettings().value("window/geometry") is not None


def test_the_navigation_folds_to_its_strip_and_stays_folded(tmp_path, monkeypatch):
    """The rail folds the way every side panel does (`gui/grip_splitter.py`): its grip,
    double-clicked or dragged past its minimum, shuts it to a strip reading "Navigation";
    remembered across sessions; Ctrl+\\ toggles it from anywhere. There is no hide button.
    """
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtGui import QKeySequence, QShortcut
    from PyQt6.QtTest import QTest

    width, height = _a_size_this_screen_can_hold()
    window = _window(tmp_path, monkeypatch)
    window.show()
    window.resize(width, height)
    window._switch_mode("tools")  # a mode with a rail
    QApplication.processEvents()
    split = window._split
    assert split.foldable_panes() == {0: "Navigation"}
    open_width = split.sizes()[0]
    assert open_width > 100
    handle = split.handle(1)
    start = handle.rect().center()
    end = QPoint(10 - handle.x(), start.y())
    QTest.mousePress(
        handle, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start
    )
    QTest.mouseMove(handle, end)
    QTest.mouseRelease(
        handle, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end
    )
    QApplication.processEvents()
    assert window.nav_hidden()
    assert split.sizes()[0] == 0, "the rail is shut"
    assert handle.width() == split.strip_width(), "a strip, never nothing"
    assert not window._tree.isHidden(), "the mode still has its rail; it is folded"
    window.close()

    again = _window(tmp_path, monkeypatch)
    again.show()
    again.resize(width, height)
    again._switch_mode("tools")
    QApplication.processEvents()
    assert again.nav_hidden(), "shut, as it was left"
    shortcuts = [
        s for s in again.findChildren(QShortcut) if s.key() == QKeySequence("Ctrl+\\")
    ]
    assert len(shortcuts) == 1
    shortcuts[0].activated.emit()
    QApplication.processEvents()
    assert not again.nav_hidden()
    assert abs(again._split.sizes()[0] - open_width) <= 2, "back at the width it had"
    shortcuts[0].activated.emit()
    QApplication.processEvents()
    assert again.nav_hidden(), "Ctrl+\\ shuts it again"
    again.close()


def test_an_old_hidden_navigation_comes_back_shut(tmp_path, monkeypatch):
    """`window/nav_hidden` and `window/nav_width` from the build with the ‹ / › button carry
    into the one scheme once, and are removed."""
    QSettings().setValue("window/nav_hidden", True)
    QSettings().setValue("window/nav_width", 210)
    width, height = _a_size_this_screen_can_hold()
    window = _window(tmp_path, monkeypatch)
    window.show()
    window.resize(width, height)
    window._switch_mode("tools")
    QApplication.processEvents()
    assert window.nav_hidden()
    assert QSettings().value("window/nav_hidden") is None
    window._split.toggle(0)
    QApplication.processEvents()
    assert abs(window._split.sizes()[0] - 210) <= 2
    window.close()


def test_a_mode_without_a_rail_shows_no_strip(tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    window._switch_mode("home")
    assert window._tree.isHidden() and window._rail_box.isHidden()
    window.close()
