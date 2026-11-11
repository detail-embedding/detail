"""A band that folds gives its space back, and is still there to open again.

The class: **a collapse that hides pixels without freeing the minimum.** Hiding a widget is only
worth doing if its floor leaves with it — otherwise the container still reserves the height, the
page is exactly as tall as before, and the control appears to work while achieving nothing. Orange3
hit this in a `QSplitter` and had to override `minimumSizeHint()` to exclude the collapsed child;
in a box layout Qt does the right thing for free, but "for free" is not "verified", and the whole
point of the band is the number.

So this asserts the **floor**, not that the child is hidden. Measured on the temporal INTERPRET
tab: 949px with both bands open, 839 with the ethograms folded, 621 with both — against the 739px
a 13" laptop can show, which is the difference between a tab that fits and one that does not.

**The fixture has to be shown.** This test was written first against an unshown workspace and
reported 949px in all three states — folding, unfolding, everything — because an unlaid-out widget
answers about a layout that has not run. It passed identically with the bands working and with them
gutted. `_floor()` below shows the widget and re-activates the layout for that reason, and the
delta assertions are what would catch it going back.

The two tests that build a *real* interpret tab live in ``test_cluster_workspace.py`` instead,
beside the ``fitted`` fixture that makes one — the same split that file already records for the
pane checks ("the corpus is wherever a surface can be built").

The other half is that a folded band is not a disappeared one. Krita removed per-pane collapse
because people could not find what they had closed; a *section* keeps its titled header, which is
what makes the fold reversible without a menu item somewhere else.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402

from detail.gui.collapse import CollapsibleBand  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_settings():
    """A band remembers its state, so a test that folds one would otherwise decide what the
    next test opens with — and the order the suite runs in would change the answers."""

    def clear():
        # One instance: `beginGroup`/`endGroup` are state *on the object*, so calling them on
        # two fresh `QSettings()` is the "No matching beginGroup" warning and clears nothing.
        settings = QSettings()
        settings.beginGroup("bands")
        settings.remove("")
        settings.endGroup()

    clear()
    yield
    clear()


def _tall(height: int = 200) -> QWidget:
    body = QLabel("content")
    body.setMinimumHeight(height)
    return body


def _hosted(band: CollapsibleBand) -> CollapsibleBand:
    band.resize(400, 400)
    band.show()
    QApplication.processEvents()
    return band


def _floor(widget) -> int:
    """The widget's minimum after the layout has actually been run.

    Not ``minimumSizeHint()`` on its own: it is answered from a cached layout, so a widget that
    was never shown gives the same number before and after a fold.
    """
    widget.layout().invalidate()
    widget.updateGeometry()
    for _ in range(3):
        QApplication.processEvents()
    return widget.minimumSizeHint().height()


def test_folding_a_band_gives_its_height_back(qapp):
    """The number the band exists for. Hiding a child that keeps its floor is a no-op with a
    triangle on it."""
    band = _hosted(CollapsibleBand("Frames", _tall(200), key="t/frames"))
    open_floor = _floor(band)

    band.set_open(False)
    folded_floor = _floor(band)

    assert folded_floor < open_floor - 150, (
        f"folding took the floor from {open_floor}px to {folded_floor}px — the content is "
        "hidden but its height is still reserved, so nothing was gained"
    )


def test_reopening_restores_the_height(qapp):
    """The negative case: a fold that cannot be undone is a delete."""
    band = _hosted(CollapsibleBand("Frames", _tall(200), key="t/frames"))
    open_floor = _floor(band)
    band.set_open(False)
    band.set_open(True)
    assert _floor(band) == open_floor


def test_a_folded_band_still_shows_its_header(qapp):
    """Sections, not panes. A collapsed pane is an unlabelled sliver nobody can find again —
    which is why Krita removed theirs."""
    band = _hosted(CollapsibleBand("Syllable ethograms", _tall(), key="t/e"))
    band.set_open(False)
    assert band._header.isVisible()
    assert "Syllable ethograms" in band._header.text()
    assert "▶" in band._header.text(), "the glyph must say which way it will go"
    band.set_open(True)
    assert "▼" in band._header.text()


def test_a_band_remembers_being_folded(qapp):
    """The page is rebuilt on every navigation, so a fold that reset each time is a control
    nobody can rely on."""
    first = _hosted(CollapsibleBand("Frames", _tall(), key="t/frames"))
    first.set_open(False)

    second = CollapsibleBand("Frames", _tall(), key="t/frames")
    assert not second.is_open(), "a rebuilt band opened against the remembered state"


def test_a_band_with_no_remembered_state_uses_its_default(qapp):
    """Both result bands open by default: the frames are how a syllable is recognised, and a
    screen that opens having hidden its own evidence is worse than one that scrolls."""
    assert CollapsibleBand("Frames", _tall(), key="t/never-set").is_open()
    assert not CollapsibleBand(
        "Frames", _tall(), key="t/never-set-2", open_=False
    ).is_open()


def test_one_bands_state_does_not_leak_into_another(qapp):
    """Keyed per band, or folding the frames would fold the ethograms with them."""
    _hosted(CollapsibleBand("Frames", _tall(), key="t/frames")).set_open(False)
    assert CollapsibleBand("Ethograms", _tall(), key="t/ethograms").is_open()


# --- the drag bar ---------------------------------------------------------------------------


def test_a_band_given_a_resize_callback_can_be_dragged(qapp):
    """Every section folds *and* drags. `ResizeHandle`'s own docstring has said since it was
    written that "a band you can drag beside a band you cannot is the thing that gets reported
    as a bug" — and that is exactly what was reported, on a screen where two of five bands had
    one.
    """
    from detail.gui.sizing import ResizeHandle

    moved: list[int] = []
    band = _hosted(
        CollapsibleBand("Clips", _tall(200), key="t/c", on_resize=moved.append)
    )
    assert isinstance(band._handle, ResizeHandle)

    band._handle.dragged.emit(40)
    assert moved == [40]


def test_a_band_with_no_resize_callback_has_no_bar(qapp):
    """A handle on a section whose content has no size to trade would be a control that does
    nothing — the promise-without-a-behaviour the map's pointing hand already taught us.
    """
    band = _hosted(CollapsibleBand("Ethograms", _tall(200), key="t/e"))
    assert band._handle is None


def test_the_bar_folds_away_with_the_section(qapp):
    """A drag bar under a folded section resizes nothing, and it would sit against the *next*
    band's header where it reads as belonging to that one.

    `isHidden`, not `isVisible`: an unshown ancestor makes everything invisible, which is the
    idiom this suite records as passing against its own bug.
    """
    band = _hosted(
        CollapsibleBand("Clips", _tall(200), key="t/c", on_resize=lambda _d: None)
    )
    assert not band._handle.isHidden()

    band.set_open(False)
    assert band._handle.isHidden()

    band.set_open(True)
    assert not band._handle.isHidden()


def test_the_bar_draws_a_grip_rather_than_a_plain_rule(qapp):
    """It has to *look* draggable, or it reads as one more separator.

    Painted with `CE_Splitter`, the primitive `QSplitterHandle` uses, so it is the same mark as
    the divider between the 3-D scatter and the cluster table — which is the one the user named.
    Asserted on rendered pixels because "it is a QFrame with no shape" would pass over a handle
    that draws nothing at all.

    **Under the app's style, because the grip is the style's.** The `qapp` fixture installs it
    (`gui/style.py`); under the platform default `CE_Splitter` paints one flat colour, and this
    test once read "no grip" against a handle that has one everywhere the user will ever see it.
    """
    from PyQt6.QtWidgets import QApplication

    band = _hosted(
        CollapsibleBand("Clips", _tall(200), key="t/c", on_resize=lambda _d: None)
    )
    band.resize(400, band.sizeHint().height())
    band.show()
    QApplication.processEvents()

    image = band._handle.grab().toImage()
    shades = {
        image.pixelColor(x, y).lightness()
        for y in range(image.height())
        for x in range(image.width())
    }
    assert len(shades) > 1, "the handle painted one flat colour — no grip on it"


def test_a_dragged_size_is_remembered(qapp):
    """A view preference the surface cannot pick on your behalf, like the fold beside it. At a
    hundred recordings the square size is set once and wanted every time after — and the page is
    rebuilt on every navigation, so nothing survives that is not written down.
    """
    from detail.gui.sizing import remember_size, remembered_size

    assert remembered_size("t/sizes", 165) == 165  # nothing stored yet

    remember_size("t/sizes", 240)

    assert remembered_size("t/sizes", 165) == 240


def test_the_grip_lies_flat_rather_than_standing_on_end(qapp):
    """Reported from a screenshot: the grips were a *column* of dots on a 6px bar.

    Qt's `State_Horizontal` names the **splitter's** orientation, not the handle's — a
    horizontal splitter puts panes side by side and draws a vertical handle. This one divides
    top from bottom, so the flag has to be off.

    Measured as the ink's shape: a flat grip spreads across the bar's width and a standing one
    stacks down its 6px height, so the number of distinct columns carrying ink is what tells
    them apart. Asserting "the flag is unset" would restate the implementation.
    """
    from PyQt6.QtWidgets import QApplication

    band = _hosted(
        CollapsibleBand("Clips", _tall(200), key="t/c", on_resize=lambda _d: None)
    )
    band.resize(400, band.sizeHint().height())
    band.show()
    QApplication.processEvents()

    image = band._handle.grab().toImage()
    background = image.pixelColor(0, 0).lightness()
    inked = {
        x
        for x in range(image.width())
        for y in range(image.height())
        if image.pixelColor(x, y).lightness() != background
    }

    assert len(inked) > image.height(), (
        f"the grip covers {len(inked)} columns of a {image.height()}px-tall bar — it is "
        "standing on end rather than lying flat"
    )
