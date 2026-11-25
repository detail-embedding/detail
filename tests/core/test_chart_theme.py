"""The chart palettes stay legal — measured, in both themes, against the surface they are drawn on.

**The class:** a chart colour that is chosen by eye. Nothing fails when one drifts — the plot
renders, the legend matches, and the only people who can see that two series have become one are
the readers the palette exists for. Found here as three instances at once: two copies of Okabe-Ito
in different orders (the first cluster orange on screen, blue on paper), an order whose worst
neighbouring pair measured ΔE 7.6 under deuteranopia against a target of 8, and a dark theme where
the only in-band steps put blue and sky blue on the same colour while every adjacent check passed.

So every colour in ``core/chart_theme.py`` is run through ``core/color_checks.py`` here, and each
check has a negative case — the historical failure it exists for — so a check that stops biting
fails too.
"""

import ast
import pathlib

import pytest

from detail.core import chart_theme, color_checks
from detail.core.chart_theme import DARK, LIGHT, THEMES
from detail.core.color_checks import (
    CVD_TARGET,
    NORMAL_FLOOR,
    check_categorical,
    check_ramp,
    contrast,
    delta_e,
    nearest_pair,
    oklch,
)

BOTH = pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])

#: Light-theme slots below 3:1 on white, and why that is allowed. Okabe-Ito's amber and sky blue
#: are 2.2-2.3:1 on white — the canonical values every reader of a figure recognises — and the
#: relief rule applies: an exported figure ships its tidy CSV beside it, and a GUI plot is read
#: with its table. A *new* colour below 3:1 is not in this set and fails.
_LIGHT_RELIEF = {"#E69F00", "#56B4E9"}


# --------------------------------------------------------------------------------------------- #
# The palettes
# --------------------------------------------------------------------------------------------- #


@BOTH
def test_the_categorical_palette_passes_every_gate(theme) -> None:
    report = check_categorical(
        theme.categorical, theme=theme.name, surface=theme.surface
    )
    assert report.failures == []
    assert report.cvd.delta_e >= CVD_TARGET, f"CVD in the floor band: {report.cvd}"


@BOTH
def test_the_first_three_slots_pass_all_pairs_for_scatter(theme) -> None:
    report = check_categorical(
        theme.categorical[:3], theme=theme.name, surface=theme.surface, scope="all"
    )
    assert report.failures == []
    assert report.cvd.delta_e >= CVD_TARGET


@BOTH
def test_no_two_slots_anywhere_are_near_duplicates(theme) -> None:
    pair = nearest_pair(theme.categorical)
    assert pair.delta_e >= NORMAL_FLOOR, pair


@BOTH
def test_the_extension_keeps_every_identity_apart(theme) -> None:
    """The frozen 24 were generated at a worst pair of ΔE 9.3; an edit that collapses two fails."""
    pair = nearest_pair(theme.categorical + theme.extended)
    assert pair.delta_e >= 9.0, pair


def test_low_contrast_slots_are_exactly_the_documented_relief() -> None:
    light = check_categorical(LIGHT.categorical, theme="light", surface=LIGHT.surface)
    dark = check_categorical(DARK.categorical, theme="dark", surface=DARK.surface)
    assert {c for c, _ in light.low_contrast} == _LIGHT_RELIEF
    assert dark.low_contrast == ()


def test_both_themes_are_one_palette_stepped_not_two() -> None:
    """Slot *i* is the same hue in both themes, so a behavior keeps its colour across themes."""
    for index, (light, dark) in enumerate(
        zip(LIGHT.categorical + LIGHT.extended, DARK.categorical + DARK.extended)
    ):
        drift = abs(oklch(light)[2] - oklch(dark)[2])
        assert min(drift, 360 - drift) <= 15, (index, light, dark)
    assert len(LIGHT.categorical) == len(DARK.categorical)
    assert len(LIGHT.extended) == len(DARK.extended)


@BOTH
def test_the_ink_reads_on_the_surface(theme) -> None:
    assert contrast(theme.text, theme.surface) >= 7.0
    assert contrast(theme.summary, theme.surface) >= 7.0
    assert contrast(theme.text_secondary, theme.surface) >= 4.5
    assert contrast(theme.text_muted, theme.surface) >= 4.5
    assert contrast(theme.axis, theme.surface) >= 3.0
    assert contrast(theme.reference, theme.surface) >= 3.0


@BOTH
def test_a_handle_stands_apart_from_the_data_and_the_ink(theme) -> None:
    """A dragged cutoff must read as a control on any plot: visible, and not a series or a summary."""
    assert contrast(theme.handle, theme.surface) >= 3.0
    assert delta_e(theme.handle, theme.identity(0)) >= NORMAL_FLOOR
    assert delta_e(theme.handle, theme.summary) >= NORMAL_FLOOR
    assert delta_e(theme.handle, theme.reference) >= NORMAL_FLOOR


@BOTH
def test_the_grid_and_the_neutral_recede(theme) -> None:
    """A gridline is visible and quiet; the neutral is quieter than any series but still a mark."""
    assert 1.1 <= contrast(theme.grid, theme.surface) <= 1.5
    quietest_series = min(contrast(c, theme.surface) for c in theme.categorical)
    assert 1.5 <= contrast(theme.neutral, theme.surface) < quietest_series
    for series in theme.categorical:
        assert delta_e(theme.neutral, series) >= NORMAL_FLOOR, series


@BOTH
def test_the_ramps_read_as_ramps(theme) -> None:
    assert check_ramp(theme.sequential) == []
    middle = len(theme.diverging) // 2
    assert check_ramp(theme.diverging[: middle + 1]) == []
    assert check_ramp(theme.diverging[middle:]) == []
    assert oklch(theme.diverging[middle])[1] < 0.02, "the midpoint must be grey"


#: How far apart the diverging scale's two poles must stay under red-green colour-vision
#: deficiency, and how much of that must be lightness. The blue ↔ vermillion pair this replaced
#: measured ΔE 18–21 with the poles at one lightness (ΔL 0.02 dark, 0.07 light): which way a
#: cluster went was hue alone, the part a colour-blind reader loses (user, 2026-11-16).
_POLE_CVD_FLOOR = 30.0
_POLE_MIN_DELTA_L = 0.12
_OLD_DIVERGING_POLES = {"light": ("#025a8d", "#ab4501"), "dark": ("#7fc5fe", "#ff9a6b")}


def _pole_failures(low: str, high: str) -> list[str]:
    failures = [
        f"{vision}: ΔE {delta_e(low, high, vision):.1f}"
        for vision in ("protan", "deutan")
        if delta_e(low, high, vision) < _POLE_CVD_FLOOR
    ]
    if abs(oklch(low)[0] - oklch(high)[0]) < _POLE_MIN_DELTA_L:
        failures.append("the poles sit at one lightness")
    return failures


@BOTH
def test_the_diverging_poles_differ_by_more_than_hue(theme) -> None:
    """Less and more stay apart for a red-green colour-blind reader, in lightness as well."""
    assert _pole_failures(theme.diverging[0], theme.diverging[-1]) == []
    # The pair it replaced fails both halves, so the check can fail.
    assert len(_pole_failures(*_OLD_DIVERGING_POLES[theme.name])) == 3


@BOTH
def test_the_colormap_reads_as_a_ramp_whose_hue_moves(theme) -> None:
    """The location plots' colormap: monotone lightness and visible steps like any ramp, but
    hue moves by design, so the one-hue rule is off. A grey ramp on a busy picture of paths
    read as black and white."""
    assert check_ramp(theme.colormap, one_hue=False) == []
    assert theme.colormap != theme.sequential


@BOTH
def test_the_colormap_s_low_end_fades_into_the_surface(theme) -> None:
    """A cell an animal barely visited reads as almost nothing, and a hotspot stands out:
    the low end is near the surface and the high end the farthest from it. Viridis on white
    put its strongest colour on the least-visited cells and framed the arena's edges."""
    contrasts = [contrast(c, theme.surface) for c in theme.colormap]
    assert contrasts[0] < 1.5
    assert contrasts[-1] == max(contrasts)


@BOTH
def test_zero_on_the_sequential_scale_sits_near_the_surface(theme) -> None:
    """Low magnitude reads as "nothing": the low end is the ramp stop closest to the surface."""
    distances = [delta_e(c, theme.surface) for c in theme.sequential]
    assert distances[0] == min(distances)


def test_identity_is_stable_as_the_count_grows() -> None:
    """Colour follows the entity: the seventh identity never repaints the first six."""
    assert LIGHT.identities(6) == LIGHT.identities(30)[:6]
    assert LIGHT.identity(6) == LIGHT.extended[0]


def test_assign_identities_is_by_sorted_label_not_first_seen() -> None:
    first = chart_theme.assign_identities(["rearing", "grooming"])
    second = chart_theme.assign_identities(["grooming", "rearing", "grooming"])
    assert (
        first
        == second
        == {"grooming": LIGHT.label_identity(0), "rearing": LIGHT.label_identity(1)}
    )


def test_a_label_is_never_the_colour_of_less_or_more() -> None:
    """Blue and yellow are the diverging scale's two poles and mean less and more. A label in
    either reads as a direction on a cluster that has none (user, 2026-11-16), so the slots
    near a pole in either theme are reserved, written out in ``LABEL_RESERVED_SLOTS`` and
    recomputed here: a palette edit that moves a colour next to a pole fails this."""
    near = set()
    for theme in (LIGHT, DARK):
        poles = (theme.diverging[0], theme.diverging[-1])
        for slot, colour in enumerate(theme.categorical + theme.extended):
            if (
                min(delta_e(colour, pole) for pole in poles)
                < chart_theme.LABEL_POLE_FLOOR
            ):
                near.add(slot)
    assert near == set(chart_theme.LABEL_RESERVED_SLOTS)
    for theme in (LIGHT, DARK):
        labels = theme.label_identities()
        assert len(labels) == 12
        assert theme.hue("blue") not in labels and theme.hue("orange") not in labels
        assert theme.label_identity(0) == theme.hue("green")
        assert theme.label_identity(len(labels)) == labels[0], "wraps past the end"
        for colour in labels:
            for pole in (theme.diverging[0], theme.diverging[-1]):
                assert delta_e(colour, pole) >= chart_theme.LABEL_POLE_FLOOR, colour
    # The palette's own first slot is the "less" pole's neighbour: the check can fail.
    assert delta_e(DARK.identity(0), DARK.diverging[0]) < chart_theme.LABEL_POLE_FLOOR


def test_a_hue_is_found_by_its_name_not_its_slot() -> None:
    """A meaning-carrying colour (TP green) must survive the slots being reordered."""
    assert len(chart_theme.HUE_NAMES) == len(LIGHT.categorical)
    for name, expected_hue in (("green", 165), ("vermillion", 47), ("sky", 236)):
        for theme in (LIGHT, DARK):
            hue = oklch(theme.hue(name))[2]
            assert abs(hue - expected_hue) < 12, (theme.name, name, hue)
    with pytest.raises(ValueError, match="hue must be one of"):
        LIGHT.hue("teal")


def test_an_unknown_theme_is_refused() -> None:
    with pytest.raises(ValueError, match="chart theme must be one of"):
        chart_theme.theme("sepia")
    assert chart_theme.theme("dark") is DARK
    assert set(THEMES) == {"light", "dark"}


# --------------------------------------------------------------------------------------------- #
# The checks bite — each on the failure it was written for
# --------------------------------------------------------------------------------------------- #


def test_the_cvd_gate_would_catch_the_order_it_replaced() -> None:
    replaced = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
    report = check_categorical(replaced, theme="light", surface="#ffffff")
    assert report.cvd.delta_e < CVD_TARGET
    assert {report.cvd.first, report.cvd.second} == {"#009E73", "#CC79A7"}


def test_the_near_duplicate_gate_would_catch_blue_on_sky_blue() -> None:
    """The dark search's collapse: adjacent checks pass, the whole-palette pair does not."""
    collapsed = ("#429de0", "#c58803", "#28ae82", "#e66d21", "#c976a4", "#40a0d4")
    adjacent = check_categorical(collapsed, theme="dark", surface="#262626")
    assert adjacent.failures == []
    assert nearest_pair(collapsed).delta_e < 5


def test_the_band_and_contrast_checks_would_catch_okabe_yellow() -> None:
    report = check_categorical(("#0072B2", "#F0E442"), theme="light", surface="#ffffff")
    assert any("#F0E442" == c for c, _ in report.off_band)
    assert any("#F0E442" == c for c, _ in report.low_contrast)


def test_the_chroma_floor_would_catch_a_grey() -> None:
    report = check_categorical(("#0072B2", "#808080"), theme="light", surface="#ffffff")
    assert [c for c, _ in report.low_chroma] == ["#808080"]


def test_the_ramp_check_would_catch_a_rainbow_and_a_zigzag() -> None:
    assert any(
        "hue spread" in f
        for f in check_ramp(("#2166ac", "#1b9e77", "#e6ab02", "#d95f02"))
    )
    assert any(
        "not monotone" in f
        for f in check_ramp(("#eef6fd", "#5999d4", "#a2c8e9", "#133d70"))
    )


def test_the_colour_math_matches_known_values() -> None:
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#767676", "#ffffff") == pytest.approx(4.54, abs=0.01)
    assert delta_e("#0072B2", "#0072B2") == 0.0
    assert oklch("#ffffff")[0] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="#rrggbb"):
        color_checks.hex_to_rgb("blue")


# --------------------------------------------------------------------------------------------- #
# Pure data
# --------------------------------------------------------------------------------------------- #

_ALLOWED_IMPORTS = {"dataclasses", "itertools", "math", "detail"}


def _foreign_imports(source: str) -> set[str]:
    """Top-level packages ``source`` imports outside the stdlib-and-self allowance."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found - _ALLOWED_IMPORTS


@pytest.mark.parametrize("module", [chart_theme, color_checks])
def test_the_design_language_imports_no_drawing_library(module) -> None:
    """Every adapter imports this module, so it may import none of them (nor numpy)."""
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8-sig")
    assert _foreign_imports(source) == set()


def test_the_import_scan_would_catch_a_drawing_library() -> None:
    planted = (
        "import math\nfrom PyQt6.QtGui import QColor\nimport matplotlib.pyplot as plt\n"
    )
    assert _foreign_imports(planted) == {"PyQt6", "matplotlib"}
