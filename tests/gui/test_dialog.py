"""A dialog draws a border of its own, since it can otherwise render the same fill as the window
it opened over.

**Reported from the app**: a launch dialog and the shell behind it were the same colour edge to
edge, with nothing marking where one window ended and the next began — a platform/theme whose
window manager draws no visible frame of its own leaves the dialog with nothing at all. A light
line around the dialog is the whole fix, painted directly on it in `Dialog.paintEvent`
(`gui/dialog.py`) rather than through a wrapped `QFrame`, because a wrapping frame would be a
third, stray widget in the dialog's top-level layout and fail `a_dialog_scrolls_its_content`
(`gui/requirements.py`), which requires that layout hold only the scrolled body and the button box.

**Rendered, not merely constructed.** A dialog whose `paintEvent` override forgets to call
`super().paintEvent(event)` builds and shows exactly like a correct one — nothing about its
structure changes — so the check reads the actual pixels back rather than checking for the
presence of a method with the right name.

**And checked against palettes this suite does not run under.** The first version of this fix
drew the border in the palette's `Mid` tone and asserted only that `Mid != Window`. Both halves
passed comfortably: the suite runs `QT_QPA_PLATFORM=offscreen`, whose `fusion` palette puts `Mid`
55 points of lightness off `Window`. On the `windows11` dark palette a real user gets, `Mid` is
`#282828` against a `#1e1e1e` window — a gap of 10, which is not a visible line. The guard could
not catch that, because the only palette it ever saw was the one where the colour was fine, and
an inequality is satisfied by any gap at all. So the contrast check below runs against
constructed palettes rather than the live one, and asserts a floor rather than difference.
"""

import pytest

from detail.gui.dialog import (
    MIN_BORDER_CONTRAST,
    Dialog,
    border_color,
)
from detail.gui.qt import QColor, QDialogButtonBox, QLabel, QPalette, QPixmap
from tests.gui.test_dialog_space import BUILDABLE, _build


def _rgb(color) -> tuple[int, int, int]:
    """A `QColor` reduced to its 8-bit channels.

    **Not `QColor.__eq__`.** A colour read back from a rendered pixmap and the same colour read
    from a palette report identical `.name()` / `.getRgb()` 8-bit values and still compare unequal
    — `QColor`'s `==` resolves through 16-bit-precision internals that can drift a handful of ULPs
    between the two paths on this Qt build, even at identical 8-bit colour. The border this test
    exists to catch is an 8-bit, human-visible difference, so that is the precision to compare at.
    """
    r, g, b, _a = color.getRgb()
    return r, g, b


def _corner_color(dialog):
    """Render `dialog` off-screen and return its top-left corner pixel."""
    dialog.resize(max(dialog.width(), 200), max(dialog.height(), 200))
    pixmap = QPixmap(dialog.size())
    dialog.render(pixmap)
    return pixmap.toImage().pixelColor(0, 0)


def _palette(window: str, ink: str, mid: str) -> QPalette:
    """A palette carrying real measured values for the roles a border might be drawn from.

    `Mid` is set explicitly, not left to Qt. A default-constructed `QPalette` derives its
    shade roles from its *own* Window brush, so setting only Window and WindowText leaves
    `Mid` at an unrelated default — and a guard built on that cannot reproduce the shipped
    bug, which was precisely a bad relationship between `Window` and `Mid` on one real style.
    (Confirmed the hard way: an earlier draft of this fixture passed with the broken
    implementation still in place.)
    """
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(window))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(ink))
    palette.setColor(QPalette.ColorRole.Mid, QColor(mid))
    return palette


#: (Window, WindowText, Mid) measured off the two styles this app actually renders under, plus
#: the extremes. `windows11-dark` is the case that shipped broken — it is here so this file
#: fails on that regression without needing a Windows box to run on. `pure-black` is here for
#: the other tempting implementation: a multiplicative `lighter()` tint, which cannot move off
#: black at all.
_PALETTES = {
    "windows11-dark": ("#1e1e1e", "#ffffff", "#282828"),
    "fusion-light": ("#efefef", "#000000", "#b8b8b8"),
    "pure-black": ("#000000", "#ffffff", "#0a0a0a"),
    "pure-white": ("#ffffff", "#000000", "#f5f5f5"),
}


@pytest.mark.parametrize("name", sorted(_PALETTES))
def test_the_border_clears_the_contrast_floor(name):
    """The border is visibly off the window fill on every palette the app renders under.

    Guards the guard: a border the same colour as what it sits on is not a border, and the
    rendering checks below would still pass, since they only ask that the pixel match whatever
    colour the code chose.
    """
    window_hex, ink_hex, mid_hex = _PALETTES[name]
    window = QColor(window_hex)
    border = border_color(_palette(window_hex, ink_hex, mid_hex))
    gap = abs(border.lightness() - window.lightness())
    assert gap >= MIN_BORDER_CONTRAST, (
        f"on the {name} palette the border is {border.name()} against a {window.name()} "
        f"window — {gap} points of lightness, under the {MIN_BORDER_CONTRAST} floor, so it "
        f"would not read as a line"
    )


def test_a_dialog_paints_a_visible_border(qapp):
    """The base case: a minimal dialog's corner pixel is the derived border colour."""
    dialog = Dialog()
    dialog.body().addWidget(QLabel("a row"))
    dialog.set_buttons(QDialogButtonBox(QDialogButtonBox.StandardButton.Ok))

    corner = _corner_color(dialog)
    assert _rgb(corner) == _rgb(border_color(dialog.palette())), (
        f"the dialog's corner pixel is {corner.name()}, not the derived border colour — "
        f"`Dialog.paintEvent` did not draw a border"
    )


@pytest.mark.parametrize("cls", BUILDABLE, ids=lambda c: c.__name__)
def test_every_dialog_in_the_app_paints_the_border(qapp, tmp_path, cls):
    """Every real dialog inherits the border — catches a subclass whose own `paintEvent` shadows
    the base without chaining to it, which building or showing the dialog alone would not.
    """
    dialog = _build(cls, tmp_path)
    try:
        corner = _corner_color(dialog)
        assert _rgb(corner) == _rgb(border_color(dialog.palette())), (
            f"{cls.__name__}'s corner pixel is {corner.name()}, not the derived border colour "
            f"— its `paintEvent` override must call `super().paintEvent(event)`"
        )
    finally:
        dialog.close()
