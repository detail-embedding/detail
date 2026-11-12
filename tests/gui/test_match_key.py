"""The match key at the top of the filter dock: a fixed 0–100% ramp that never stretches to
the data."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QColor  # noqa: E402

from detail.core.chart_theme import DARK, LIGHT  # noqa: E402
from detail.gui import match_key as mk  # noqa: E402
from detail.gui import plot_theme  # noqa: E402


@pytest.fixture
def key(qapp):
    widget = mk.MatchKey()
    widget.resize(600, 30)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()


def _fraction_along(ramp, share: float) -> float:
    left, right = ramp.x_for(0.0), ramp.x_for(1.0)
    return (ramp.x_for(share) - left) / (right - left)


def _ink_columns(ramp, top: float, bottom: float) -> set[int]:
    """The x of every column holding the theme's text colour between ``top`` and ``bottom``."""
    image = ramp.grab().toImage()
    ratio = image.devicePixelRatio()
    ink = QColor(plot_theme.current().text)
    out = set()
    for x in range(ramp.width()):
        for y in range(int(top), int(bottom)):
            if _close(image.pixelColor(int(x * ratio), int(y * ratio)), ink, 10):
                out.add(x)
                break
    return out


def test_the_scale_is_fixed_and_its_ends_are_the_ramps(key):
    """43% sits at 43% of the bar whatever else is shown; 0% and 100% are the ramp's stops."""
    key.set_filter(True, selected_share=0.43, run_share=0.49)
    ramp = key.ramp
    assert _fraction_along(ramp, 0.43) == pytest.approx(0.43)
    image = ramp.grab().toImage()
    y = mk._BAR_TOP + mk._BAR_H // 2
    low = image.pixelColor(int(ramp.x_for(0.0)) + 2, y)
    high = image.pixelColor(int(ramp.x_for(1.0)) - 2, y)
    theme = plot_theme.current()
    assert _close(low, QColor(theme.sequential[0]))
    assert _close(high, QColor(theme.sequential[-1]))


def test_the_selected_cluster_is_a_tall_bar_with_its_share_beside_it(key):
    """The selected mark spans the ramp's height and carries ``selected 2.3%``; the run keeps
    its own thinner line and ``run 23%``."""
    key.set_filter(True, selected_share=0.023, run_share=0.23)
    assert key.ramp.labels() == ["selected 2.3%", "run 23%"]
    assert key.text() == "Clips that match | 0% … 100% | selected 2.3% | run 23%"
    x = round(key.ramp.x_for(0.023))
    over_the_bar = _ink_columns(key.ramp, mk._BAR_TOP, mk._BAR_TOP + mk._BAR_H)
    assert {
        x - 1,
        x,
    } <= over_the_bar, "the bar crosses the whole ramp at the cluster's share"
    key.set_filter(True, selected_share=None, run_share=0.23)
    assert not (
        {x - 1, x} & _ink_columns(key.ramp, mk._BAR_TOP, mk._BAR_TOP + mk._BAR_H)
    ), "with nothing selected there is no bar"
    assert key.ramp.labels() == ["run 23%"]


@pytest.mark.parametrize("share", [0.0, 1.0], ids=["0%", "100%"])
def test_the_labels_stay_inside_the_key_at_either_end(key, share):
    key.set_filter(True, selected_share=share, run_share=share)
    ramp = key.ramp
    for text, top in (("selected", 0), ("run", mk._RUN_TOP)):
        label = next(t for t in ramp.labels() if t.startswith(text))
        rect = ramp.label_rect(label, share, top)
        assert rect.left() >= 0 and rect.right() <= ramp.width()
        assert rect.width() > ramp.fontMetrics().horizontalAdvance(label)


def test_the_selected_and_run_labels_never_share_a_row(key):
    """At the same share the two words would overlap; they sit above and below the bar."""
    key.set_filter(True, selected_share=0.5, run_share=0.5)
    ramp = key.ramp
    above = ramp.label_rect("selected 50%", 0.5, 0)
    below = ramp.label_rect("run 50%", 0.5, mk._RUN_TOP)
    assert not above.intersects(below)
    assert above.bottom() <= mk._BAR_TOP <= mk._BAR_TOP + mk._BAR_H <= below.top()


def test_there_is_no_tick_per_cluster_above_the_bar(key):
    """The barcode of one tick per cluster is gone: above the bar is only the selected mark."""
    assert not hasattr(key.ramp, "_shares")
    key.set_filter(True, selected_share=None, run_share=0.4)
    image = key.ramp.grab().toImage()
    ratio = image.devicePixelRatio()
    blank = image.pixelColor(0, 0)
    marked = {
        x
        for x in range(key.ramp.width())
        for y in range(mk._BAR_TOP - 3)
        if not _close(image.pixelColor(int(x * ratio), int(y * ratio)), blank, 4)
    }
    assert not marked, "nothing is drawn above the bar"


def test_share_text_keeps_a_decimal_only_where_it_matters():
    assert mk.share_text(0.023) == "2.3%"
    assert mk.share_text(0.23) == "23%"
    assert mk.share_text(1.0) == "100%"


def test_the_caption_and_scale(key):
    key.set_filter(True, selected_share=0.6, run_share=0.41)
    assert key._caption.text() == "Clips that match"
    assert key._low.text() == "0%" and key._high.text() == "100%"


def test_with_no_filter_the_key_is_greyed_with_no_marks_in_the_same_space(key, qapp):
    """User, review round 8: the key keeps its place with no filter — greyed, no marks, saying
    why — so nothing under it moves when the first filter goes on."""
    key.show()
    key.set_filter(True, selected_share=0.3, run_share=0.5)
    qapp.processEvents()
    height = key.sizeHint().height()
    key.set_filter(False)
    qapp.processEvents()
    assert not key.isHidden() and not key.is_active()
    assert key.toolTip() == mk.INACTIVE_TEXT and key.ramp.toolTip() == mk.INACTIVE_TEXT
    assert key.text() == "Clips that match | 0% … 100%", "the scale, no marks"
    assert key.sizeHint().height() == height
    key.set_filter(True, run_share=0.5)
    assert key.is_active() and key.toolTip() == ""
    assert key.text() == "Clips that match | 0% … 100% | run 50%"


def test_the_ramp_does_not_grow_with_the_window(key, qapp):
    key.resize(1600, 30)
    qapp.processEvents()
    assert key.ramp.width() == mk.RAMP_PX


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_match_colour_is_the_fixed_sequential_ramp_with_grey_for_none(qapp, theme):
    assert plot_theme.match_colour(0.0, theme) == QColor(theme.neutral)
    assert plot_theme.match_colour(1.0, theme) == QColor(theme.sequential[-1])
    assert plot_theme.match_colour(0.5, theme) == QColor(theme.sequential[4])
    assert _close(plot_theme.match_colour(1e-9, theme), QColor(theme.sequential[0]))


def test_swatches_show_each_label_with_its_time(qapp):
    row = mk.TreeKeyRow()
    row.swatches.set_labels(
        [("rearing", "#3a78c2", 252.0), ("grooming", "#c2913a", 61)]
    )
    assert row.swatches.texts() == ["■ rearing 4m12s", "■ grooming 1m01s"]
    row.swatches.set_labels([("rearing", "#3a78c2", 1.0)])
    assert row.swatches.texts() == ["■ rearing 1s"]


def _close(a: QColor, b: QColor, tolerance: int = 6) -> bool:
    return all(
        abs(x - y) <= tolerance
        for x, y in zip((a.red(), a.green(), a.blue()), (b.red(), b.green(), b.blue()))
    )


# --------------------------------------------------------------------------- #
# the effect mode: the key reads a screen's statistic while the screen is the only filter
# --------------------------------------------------------------------------- #


def test_effect_position_is_one_fixed_map_with_the_floor_as_neutral():
    from detail.gui.match_key import LOG_RATIO_EXTENT, effect_extent, effect_position

    assert effect_extent("trend") == 1.0
    assert effect_extent("paired") == LOG_RATIO_EXTENT == math.log(20.0)
    assert effect_position(None, True, 1.0) != effect_position(None, True, 1.0), "NaN"
    assert effect_position(0.9, False, 1.0) == 0.0, "inside the floor is neutral"
    assert effect_position(-0.74, True, 1.0) == pytest.approx(-0.74)
    assert effect_position(math.log(12.4), True, LOG_RATIO_EXTENT) == pytest.approx(
        math.log(12.4) / math.log(20.0)
    )
    assert effect_position(math.log(400.0), True, LOG_RATIO_EXTENT) == 1.0, "clamped"
    assert effect_position(-math.log(400.0), True, LOG_RATIO_EXTENT) == -1.0


def test_effect_words_read_as_a_factor_or_a_rho():
    from detail.gui.match_key import effect_axis_labels, effect_value_text

    assert effect_value_text(math.log(12.4), "paired") == "12.4×"
    assert effect_value_text(-math.log(2.1), "paired") == "÷2.1"
    assert effect_value_text(-0.74, "trend") == "ρ -0.74"
    assert effect_value_text(None, "trend") == "undefined"
    assert effect_axis_labels("trend") == ("ρ −1", "ρ +1")
    assert effect_axis_labels("paired") == ("÷20", "×20")


@pytest.mark.parametrize("theme", [LIGHT, DARK])
def test_effect_colour_is_the_fixed_diverging_scale_with_grey_in_the_middle(
    qapp, theme
):
    assert plot_theme.effect_colour(0.0, theme) == QColor(theme.neutral)
    assert plot_theme.effect_colour(float("nan"), theme) == QColor(theme.neutral)
    assert plot_theme.effect_colour(-1.0, theme) == QColor(theme.diverging[0])
    assert plot_theme.effect_colour(1.0, theme) == QColor(theme.diverging[-1])
    assert plot_theme.effect_colour(0.5, theme) == QColor(theme.diverging[6])
    assert plot_theme.effect_colour(5.0, theme) == QColor(
        theme.diverging[-1]
    ), "clamped"


def test_the_key_switches_to_the_effect_and_back(key):
    key.set_effect(
        "Less ↔ more under drug",
        "paired",
        floor=math.log(7.14),
        selected_effect=math.log(12.4),
    )
    assert key.is_effect()
    assert key.text() == "Less ↔ more under drug | ÷20 … ×20 | selected 12.4× | floor"
    assert key.ramp.x_for(0.0) == pytest.approx(key.ramp.width() / 2, abs=1)
    assert key.ramp.x_for(1.0) > key.ramp.x_for(0.5) > key.ramp.x_for(-0.5)

    key.set_effect("Less ↔ more with dose", "trend", floor=0.54, selected_effect=None)
    assert key.text() == "Less ↔ more with dose | ρ −1 … ρ +1 | floor"

    key.set_filter(True, selected_share=0.3, run_share=0.1)
    assert not key.is_effect()
    assert key.text() == "Clips that match | 0% … 100% | selected 30% | run 10%"


def test_a_profile_axis_is_one_sided_by_size(key):
    """A profile's effect belongs to a pair, so the key reads its size from 1× up with one
    floor tick, and a block's position is its magnitude whichever way the pair runs (§22).
    """
    from detail.gui.match_key import (
        LOG_RATIO_EXTENT,
        effect_axis_labels,
        effect_position,
        effect_value_text,
        one_sided,
    )

    assert one_sided("profile") and not one_sided("paired") and not one_sided("trend")
    assert effect_axis_labels("profile") == ("1×", "×20")
    assert effect_value_text(-math.log(2.1), "profile") == "2.1×"
    assert effect_position(
        -math.log(12.4), True, LOG_RATIO_EXTENT, one_sided=True
    ) == pytest.approx(math.log(12.4) / math.log(20.0))
    assert (
        effect_position(math.log(400.0), True, LOG_RATIO_EXTENT, one_sided=True) == 1.0
    )
    assert effect_position(0.9, False, LOG_RATIO_EXTENT, one_sided=True) == 0.0

    key.set_effect(
        "Differs across session",
        "profile",
        floor=math.log(4.0),
        selected_effect=-math.log(12.4),
    )
    assert key.is_effect()
    assert key.text() == "Differs across session | 1× … ×20 | selected 12.4× | floor"
    ramp = key.ramp
    assert ramp.x_for(0.0) == pytest.approx(ramp.x_for(-1.0)), "no negative half"
    assert ramp.x_for(0.0) < ramp.x_for(0.5) < ramp.x_for(1.0)
    image = ramp.grab().toImage()
    y = mk._BAR_TOP + mk._BAR_H // 2
    theme = plot_theme.current()
    middle = theme.diverging[len(theme.diverging) // 2]
    assert _close(image.pixelColor(int(ramp.x_for(0.0)) + 2, y), QColor(middle))
    assert _close(
        image.pixelColor(int(ramp.x_for(1.0)) - 2, y), QColor(theme.diverging[-1])
    )

    key.set_effect("Less ↔ more under drug", "paired", floor=0.5, selected_effect=0.2)
    assert ramp.x_for(0.0) == pytest.approx(ramp.width() / 2, abs=1), "two-sided again"
