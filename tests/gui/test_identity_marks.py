"""An identity is never told apart by hue alone: a colour and a pattern, readable in greyscale.

**The class.** Two things on screen that stand for different identities — two behavior labels,
two clusters — differing only by hue. BROWSE drew each label's ethogram bands, and the swatch
keying them in its label rail and its list, in the label's colour and nothing else. To the user
who asked for this (colour-blind, 2026-11-23), "groom" and "hi" were the same grey, and the
key could not say which band was which.

**How it hid.** Every test that looked at those colours asked for them by name —
``label_identity(slot).name()`` — which checks the property a reader with full colour vision
uses, and the one that was never in doubt. Nothing rendered a mark and looked at it the way the
reader who had the problem sees it.

**The rule.** A label slot is a colour and a pattern from one mapping (``plot_theme.label_mark``),
drawn the same way in a band and in the key to it (``plot_theme.paint_label_mark``,
``plot_theme.label_swatch``). Slot 0 is solid, so a one-behavior run looks as it always did. There
is one pattern per label colour, so the two wrap together at the limit the colours already had.

**The guard.** Each test renders marks, takes the hue away — draws every slot in one grey, or
reads the real colours as luminance — and asserts that every pair still reads as two by its
**texture** (``tests/gui/greyscale.py``), never by lightness alone. The negative case shows the
measure rejecting colour alone: two solid fills of one lightness, and one pattern in two hues.

The source half — no swatch placed as a bare colour — is
``requirements.an_identity_swatch_is_never_colour_alone``, swept by
``tests/gui/test_requirements.py``; the surfaces still drawn in colour alone are listed there,
each with why.
"""

import pytest

from detail.core.chart_theme import DARK, LIGHT
from detail.gui import plot_theme
from detail.gui.cluster_strip import EthogramStrip, cluster_colors
from detail.gui.qt import QColor, QImage, QPainter, QRectF
from tests.gui import greyscale

#: Every slot with a colour of its own.
SLOTS = range(len(LIGHT.label_identities()))

#: Far enough to find a pattern's phase: its widest period is under 10px.
SHIFT = 8

#: One grey for every slot — the hue taken away entirely, not just converted.
GREY = QColor(128, 128, 128)


def _mark(slot: int, color: QColor | None, theme, width: int = 64) -> QImage:
    image = QImage(width, 16, QImage.Format.Format_ARGB32)
    image.fill(QColor(theme.surface))
    painter = QPainter(image)
    plot_theme.paint_label_mark(
        painter, QRectF(0, 0, width, 16), slot, color=color, theme=theme
    )
    painter.end()
    return image


def test_the_greyscale_measure_rejects_hue_alone(qapp):
    """The counterexample: colour-only marks must read as **one**, or every check below is
    vacuous. Two solid fills of one lightness, and one pattern drawn in two hues."""
    pink, green = QColor(204, 121, 167), QColor(0, 171, 0)
    solid = {}
    for name, colour in (("pink", pink), ("green", green)):
        image = QImage(16, 16, QImage.Format.Format_ARGB32)
        image.fill(colour)
        solid[name] = greyscale.luminance(image)
    assert pink.hue() != green.hue()
    assert abs(solid["pink"].mean() - solid["green"].mean()) < 1, "one lightness"
    assert not greyscale.apart(solid["pink"], solid["green"])

    one_pattern = [greyscale.luminance(_mark(1, c, LIGHT)) for c in (pink, green)]
    assert not greyscale.apart(*one_pattern, shift=SHIFT), "one pattern is one identity"

    # And it is the pattern it measures: the same two hues, slots 0 and 1, read as two.
    assert greyscale.apart(
        greyscale.luminance(_mark(0, pink, LIGHT)),
        greyscale.luminance(_mark(1, green, LIGHT)),
        shift=SHIFT,
    )


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=lambda t: t.name)
def test_every_label_colour_has_a_pattern_of_its_own(theme):
    """One pattern per colour, wrapping with it — and slot 0, the one-behavior run, is solid."""
    n = len(theme.label_identities())
    assert len(plot_theme.LABEL_PATTERNS) == n
    assert len(set(plot_theme.LABEL_PATTERNS)) == n
    assert plot_theme.label_mark(0, theme)[1] == "solid"
    for slot in SLOTS:
        colour, pattern = plot_theme.label_mark(slot, theme)
        again, wrapped = plot_theme.label_mark(slot + n, theme)
        assert (colour.name(), pattern) == (again.name(), wrapped)
        assert colour.name() == plot_theme.label_identity(slot, theme).name()


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=lambda t: t.name)
@pytest.mark.parametrize("hue", ["one grey", "label colours"])
def test_every_label_mark_reads_apart_in_greyscale(qapp, theme, hue):
    marks = {
        slot: greyscale.luminance(
            _mark(slot, GREY if hue == "one grey" else None, theme)
        )
        for slot in SLOTS
    }
    assert greyscale.pairs_together(marks, shift=SHIFT) == []


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=lambda t: t.name)
def test_every_label_swatch_reads_apart_in_greyscale(qapp, theme):
    """The key: what a list shows beside a label's name, in its real colour, read as grey."""
    swatches = {
        slot: greyscale.luminance(plot_theme.label_swatch(slot, theme))
        for slot in SLOTS
    }
    assert greyscale.pairs_together(swatches) == []


@pytest.mark.parametrize("hue", ["one grey", "label colours"])
def test_the_ethograms_bands_read_apart_in_greyscale(qapp, hue):
    """Rendered by the strip itself, one wide band per slot, shown and laid out."""
    n = len(SLOTS)
    strip = EthogramStrip()
    colors = [(0.5, 0.5, 0.5, 1.0)] * n if hue == "one grey" else cluster_colors(n)
    strip.set_rows(
        [("r", [(i / n, (i + 1) / n, i) for i in SLOTS])],
        colors,
        None,
        lengths_ns=[n * 10**9],
    )
    strip.resize(strip._gutter + n * 70, strip.height())
    strip.show()
    qapp.processEvents()
    try:
        image = strip.grab().toImage()
        top = strip._bar_top(0)
        bands = {}
        for slot in SLOTS:
            x0, x1 = strip._x(slot / n, 0) + 2, strip._x((slot + 1) / n, 0) - 2
            assert x1 - x0 >= 48, "the fixture must give each band room for its pattern"
            bands[slot] = greyscale.luminance(
                image.copy(x0, top, x1 - x0, strip.BAR_PX)
            )
    finally:
        strip.close()
    assert greyscale.pairs_together(bands, shift=SHIFT) == []
    assert not greyscale.texture(bands[0]).any(), "slot 0 is solid: a clean single band"
