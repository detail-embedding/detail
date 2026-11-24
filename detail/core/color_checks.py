"""The measurable half of a chart palette — computed, never eyeballed.

A palette is legal when it passes a fixed set of checks, and every one that can be measured from
the hex values alone lives here: the OKLCH lightness band and chroma floor, separation between
slots under simulated colour-vision deficiency, the same separation under normal vision, and
WCAG contrast against the surface the marks are drawn on. ``core/chart_theme.py`` declares the
palettes; ``tests/core/test_chart_theme.py`` runs them through this module in both themes, so a
colour that drifts out of a check fails a test rather than a reader.

**The simulation model is part of the standard.** ΔE is Euclidean distance in OKLab ×100, and
the CVD thresholds (target 8, floor 6) are calibrated to the Machado–Oliveira–Fernandes (2009)
transforms at severity 1.0. Swapping in another simulation (Viénot 1999, Brettel 1997) moves
borderline pairs and would need the thresholds recalibrated — so the matrices are pinned here,
not imported.

**Adjacent pairs, all pairs, and why both.** Lines, bars and stacks only put neighbouring slots
next to each other, so the CVD and normal-vision gates run over adjacent pairs of the fixed order.
A scatter can put any two marks side by side, so its forms are capped at the slots that pass
*all* pairs. A third gate, :func:`nearest_pair`, is this project's own: it runs over every pair
of the whole palette under normal vision. The adjacent gates cannot see two non-neighbouring slots
that have collapsed onto one colour — which is exactly what the dark theme did to Okabe-Ito's blue
and sky blue (one hue, separated only by lightness) when an early search pushed both to the same
lightness step, and every adjacent check still passed.
"""

import math
from dataclasses import dataclass
from itertools import combinations

#: OKLCH lightness band a categorical mark must sit in, per theme. The reference method caps the
#: dark band at 0.67; this project uses 0.77 in both, because Okabe-Ito's amber and sky blue sit
#: at 0.74-0.75 and a 0.67 cap turns them olive and ochre and drops sky blue onto blue — measured
#: while choosing the dark steps (``docs/chart_design.md``).
LIGHTNESS_BAND: dict[str, tuple[float, float]] = {
    "light": (0.43, 0.77),
    "dark": (0.48, 0.77),
}
#: Below this OKLCH chroma a hue reads as grey and stops carrying identity.
CHROMA_FLOOR = 0.10
#: Worst-pair ΔE under protan / deutan simulation: the target, and the floor that is legal only
#: with a second encoding (direct labels, gaps, texture).
CVD_TARGET = 8.0
CVD_FLOOR = 6.0
#: Worst-pair ΔE under normal vision. A hard gate — a second encoding does not excuse it.
NORMAL_FLOOR = 15.0
#: WCAG contrast a mark needs against its surface. Below it is legal only with the values readable
#: another way (a visible label, the exported table).
MARK_CONTRAST = 3.0
#: An ordered ramp: each step at least this far in OKLCH L from its neighbour, one hue within
#: this many degrees.
RAMP_MIN_DELTA_L = 0.06
RAMP_MAX_HUE_SPREAD = 40.0

# Machado, Oliveira & Fernandes (2009) CVD transforms at severity 1.0, applied in linear RGB.
_MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` → ``(r, g, b)`` in 0-255.

    Raises:
        ValueError: If ``color`` is not a six-digit hex colour.
    """
    digits = color.strip().removeprefix("#")
    if len(digits) != 6 or any(c not in "0123456789abcdefABCDEF" for c in digits):
        raise ValueError(f"expected a #rrggbb colour, got {color!r}")
    return tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))


def _linear(color: str) -> tuple[float, float, float]:
    def channel(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return tuple(channel(v) for v in hex_to_rgb(color))


def _oklab_from_linear(rgb) -> tuple[float, float, float]:
    r, g, b = rgb
    l_ = math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def oklch(color: str) -> tuple[float, float, float]:
    """``(L, C, h°)`` of a hex colour in OKLCH."""
    lightness, a, b = _oklab_from_linear(_linear(color))
    return lightness, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def relative_luminance(color: str) -> float:
    """WCAG relative luminance of a hex colour."""
    r, g, b = _linear(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two hex colours (1-21)."""
    high, low = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _simulate(color: str, kind: str) -> tuple[float, float, float]:
    matrix = _MACHADO[kind]
    rgb = _linear(color)
    return tuple(
        min(1.0, max(0.0, sum(m * c for m, c in zip(row, rgb)))) for row in matrix
    )


def delta_e(a: str, b: str, vision: str | None = None) -> float:
    """OKLab ΔE ×100 between two colours, as seen with ``vision`` (``None`` = normal)."""
    if vision is None:
        first, second = _linear(a), _linear(b)
    else:
        first, second = _simulate(a, vision), _simulate(b, vision)
    return 100 * math.dist(_oklab_from_linear(first), _oklab_from_linear(second))


@dataclass(frozen=True)
class PairResult:
    """The closest pair a check found, and how close."""

    delta_e: float
    first: str
    second: str
    vision: str

    def __str__(self) -> str:
        return f"{self.first}↔{self.second} ΔE {self.delta_e:.1f} ({self.vision})"


def _pairs(n: int, scope: str) -> list[tuple[int, int]]:
    if scope == "adjacent":
        return [(i, i + 1) for i in range(n - 1)]
    if scope == "all":
        return list(combinations(range(n), 2))
    raise ValueError(f"scope must be 'adjacent' or 'all', got {scope!r}")


def worst_cvd_pair(palette, scope: str = "adjacent") -> PairResult:
    """The closest pair under protan or deutan simulation. Tritan, far rarer, is not gated."""
    return min(
        (
            PairResult(
                delta_e(palette[i], palette[j], kind), palette[i], palette[j], kind
            )
            for kind in ("protan", "deutan")
            for i, j in _pairs(len(palette), scope)
        ),
        key=lambda r: r.delta_e,
    )


def worst_normal_pair(palette, scope: str = "adjacent") -> PairResult:
    """The closest pair under normal vision."""
    return min(
        (
            PairResult(
                delta_e(palette[i], palette[j]), palette[i], palette[j], "normal"
            )
            for i, j in _pairs(len(palette), scope)
        ),
        key=lambda r: r.delta_e,
    )


def nearest_pair(palette) -> PairResult:
    """The two slots anywhere in the palette that are hardest to tell apart (normal vision)."""
    return worst_normal_pair(palette, scope="all")


@dataclass(frozen=True)
class PaletteReport:
    """Every categorical check for one palette on one surface. ``failures`` empty = legal."""

    theme: str
    surface: str
    off_band: tuple[tuple[str, float], ...]
    low_chroma: tuple[tuple[str, float], ...]
    cvd: PairResult
    normal: PairResult
    low_contrast: tuple[tuple[str, float], ...]

    @property
    def failures(self) -> list[str]:
        """The hard failures, in words. Low contrast is relief, not failure (see module doc)."""
        out = []
        if self.off_band:
            out.append(f"outside the {self.theme} lightness band: {self.off_band}")
        if self.low_chroma:
            out.append(f"below the chroma floor (reads grey): {self.low_chroma}")
        if self.cvd.delta_e < CVD_FLOOR:
            out.append(f"CVD separation below {CVD_FLOOR}: {self.cvd}")
        if self.normal.delta_e < NORMAL_FLOOR:
            out.append(f"normal-vision separation below {NORMAL_FLOOR}: {self.normal}")
        return out


def check_categorical(
    palette, *, theme: str, surface: str, scope: str = "adjacent"
) -> PaletteReport:
    """Run the categorical checks over ``palette`` drawn on ``surface`` in ``theme``."""
    low, high = LIGHTNESS_BAND[theme]
    return PaletteReport(
        theme=theme,
        surface=surface,
        off_band=tuple(
            (c, round(oklch(c)[0], 3))
            for c in palette
            if not low <= oklch(c)[0] <= high
        ),
        low_chroma=tuple(
            (c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR
        ),
        cvd=worst_cvd_pair(palette, scope),
        normal=worst_normal_pair(palette, scope),
        low_contrast=tuple(
            (c, round(contrast(c, surface), 2))
            for c in palette
            if contrast(c, surface) < MARK_CONTRAST
        ),
    )


def check_ramp(ramp, *, one_hue: bool = True) -> list[str]:
    """The failures of an ordered ramp: monotone lightness, visible steps, and (unless
    ``one_hue`` is False, for a perceptual colormap like viridis whose hue moves by design)
    one hue.

    A correct ramp fails the categorical checks by design — it spans the lightness band and its
    pale end drops below the chroma floor — so an ordered scale is checked for reading *as* a
    ramp instead. The near-neutral end is excluded from the hue check, since a hue angle means
    nothing at zero chroma.
    """
    failures = []
    lightness = [oklch(c)[0] for c in ramp]
    rising = all(a < b for a, b in zip(lightness, lightness[1:]))
    falling = all(a > b for a, b in zip(lightness, lightness[1:]))
    if not (rising or falling):
        failures.append(
            f"lightness is not monotone: {[round(v, 3) for v in lightness]}"
        )
    thin = [
        (ramp[i], ramp[i + 1], round(abs(b - a), 3))
        for i, (a, b) in enumerate(zip(lightness, lightness[1:]))
        if abs(b - a) < RAMP_MIN_DELTA_L
    ]
    if thin:
        failures.append(f"steps closer than ΔL {RAMP_MIN_DELTA_L}: {thin}")
    hues = [oklch(c)[2] for c in ramp if oklch(c)[1] >= 0.03] if one_hue else []
    if hues:
        spread = max(hues) - min(hues)
        spread = 360 - spread if spread > 180 else spread
        if spread > RAMP_MAX_HUE_SPREAD:
            failures.append(f"hue spread {spread:.0f}° is more than one hue")
    return failures
