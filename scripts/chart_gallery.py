#!/usr/bin/env python3
"""Render every chart form in the design language, in both themes, for a person to look at.

    python scripts/chart_gallery.py --out gallery/

The validator (``core/color_checks.py``) measures colour; nothing measures whether a chart is
*good* — whether a label collides, a legend covers data, a violin reads. That takes a look, and
this is the page to look at: one sheet of swatches with the checks' numbers beside them, one sheet
of every primitive in ``report/figures.py`` on synthetic data shaped like this project's (a handful
of animals a group, heavy-tailed bout lengths, a dozen behaviors), the behavior-export figures as
they are written to disk, and — when Qt is importable — the live GUI plots on both surfaces.

Written to ``--out`` as PNG. Nothing here is read by the pipeline; it is the before/after a
change to ``core/chart_theme.py`` or ``report/figures.py`` should be reviewed against.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from detail.core import chart_theme, color_checks  # noqa: E402
from detail.report import behavior_figures, figures  # noqa: E402
from detail.report.style import FigureStyle  # noqa: E402

_NS = 1_000_000_000


def _swatches(style: FigureStyle, out: pathlib.Path) -> pathlib.Path:
    """Every colour in the theme, with the checks that constrain it."""
    import matplotlib.pyplot as plt

    colors = style.chart
    report = color_checks.check_categorical(
        colors.categorical, theme=colors.name, surface=colors.surface
    )
    scatter = color_checks.check_categorical(
        colors.categorical[:3], theme=colors.name, surface=colors.surface, scope="all"
    )
    rows = [
        ("categorical", list(colors.categorical)),
        ("extended", list(colors.extended)),
        ("sequential", list(colors.sequential)),
        ("diverging", list(colors.diverging)),
        (
            "ink",
            [
                colors.text,
                colors.text_secondary,
                colors.text_muted,
                colors.axis,
                colors.reference,
                colors.handle,
                colors.grid,
                colors.neutral,
            ],
        ),
    ]
    with figures.styled(style):
        fig, ax = plt.subplots(figsize=(style.double_col_width_in, 3.6))
        for r, (name, swatch) in enumerate(rows):
            y = len(rows) - 1 - r
            for c, color in enumerate(swatch):
                ax.add_patch(plt.Rectangle((c, y + 0.1), 0.92, 0.8, color=color, lw=0))
            ax.text(-0.3, y + 0.5, name, ha="right", va="center")
        ax.set_xlim(-2.4, 18.2)
        ax.set_ylim(-1.6, len(rows))
        ax.axis("off")
        ax.text(
            -2.4,
            -0.9,
            f"adjacent CVD ΔE {report.cvd.delta_e:.1f} · normal ΔE {report.normal.delta_e:.1f}"
            f" · first-3 all-pairs CVD {scatter.cvd.delta_e:.1f}"
            f" · nearest pair {color_checks.nearest_pair(colors.categorical).delta_e:.1f}"
            f" · below 3:1: {', '.join(c for c, _ in report.low_contrast) or 'none'}"
            f" · ramps: {_ramp_verdict(colors)}",
            fontsize=style.type.annotation,
            color=colors.text_muted,
        )
        ax.set_title(f"{colors.name} theme on {colors.surface}", loc="left")
        return figures.save_figure(fig, out / f"swatches_{colors.name}", ("png",))[0]


def _ramp_verdict(colors) -> str:
    """The ordered scales' check, in a word — "pass", or what failed."""
    middle = len(colors.diverging) // 2
    failures = [
        *color_checks.check_ramp(colors.sequential),
        *color_checks.check_ramp(colors.diverging[: middle + 1]),
        *color_checks.check_ramp(colors.diverging[middle:]),
    ]
    return "pass" if not failures else "; ".join(failures)


def _composed(style: FigureStyle, out: pathlib.Path) -> pathlib.Path:
    """A multi-panel figure as a paper would carry it: double column, lettered panels."""
    rng = np.random.default_rng(5)
    groups = ["Saline", "Cocaine"]
    fig, axes = figures.grid_multiples(
        3,
        cols=3,
        width=style.double_col_width_in,
        height_per=2.2,
        sharex=False,
        style=style,
    )
    with figures.styled(style):
        figures.observation_dots(
            axes[0],
            groups,
            [list(rng.normal(3.0, 0.5, 4)), list(rng.normal(4.1, 0.6, 4))],
            style=style,
            ylabel="rearing bouts / min",
        )
        windows = [f"{m}" for m in range(0, 30, 5)]
        figures.group_curves(
            axes[1],
            windows,
            [
                (g, list(np.round(b + np.cumsum(rng.normal(d, 0.2, 6)), 2)))
                for g, b, d in (("Saline", 3.0, 0.0), ("Cocaine", 3.3, 0.15))
            ],
            style=style,
            xlabel="minute",
            ylabel="speed (cm/s)",
        )
        figures.ranked_bars_with_floor(
            axes[2],
            ["speed", "rearing", "grooming"],
            [0.8, 0.5, 0.2],
            0.45,
            style=style,
            xlabel="|effect size|",
        )
        figures.panel_letters(axes, style=style)
        return figures.save_figure(fig, out / f"composed_{style.theme}", ("png",))[0]


def _primitive_panels(style: FigureStyle):
    """``(name, draw)`` for every chart form, on data shaped like this project's."""
    rng = np.random.default_rng(7)
    groups = ["Saline", "Cocaine", "Cocaine+Antagonist"]
    labels = ["rear", "groom", "walk", "still", "turn"]

    def dots(ax):
        small = [list(rng.normal(m, 0.6, n)) for m, n in ((3.0, 4), (4.3, 3), (3.4, 4))]
        figures.observation_dots(
            ax, groups, small, style=style, ylabel="rearing bouts / min"
        )
        figures.caption(ax, "one observation is: animal", style=style)

    def violins(ax):
        large = [list(rng.gamma(k, 0.6, 90)) for k in (3.0, 4.2, 3.5)]
        figures.group_distribution(
            ax, groups, large, style=style, ylabel="bout duration (s)"
        )

    def grouped_bars(ax):
        figures.grouped_dots(
            ax,
            ["climb", "groom", "rear"],
            [
                ("silhouette", [0.42, 0.31, 0.55], [0.03, 0.04, 0.05]),
                ("context crop", [0.55, 0.29, 0.61], [0.04, 0.05, 0.03]),
            ],
            style=style,
            ylabel="average precision",
            as_bars=True,
        )

    def curves(ax):
        windows = [f"{m}–{m + 5} min" for m in range(0, 30, 3)]
        figures.group_curves(
            ax,
            windows,
            [
                (g, list(np.round(b + np.cumsum(rng.normal(d, 0.25, len(windows))), 2)))
                for g, b, d in (("Saline", 3.0, -0.05), ("Cocaine", 3.4, 0.12))
            ],
            style=style,
            xlabel="window",
            ylabel="speed (cm/s)",
        )

    def paired(ax):
        figures.paired_lines(
            ax,
            ["Pre", "Post"],
            [(f"m{i}", list(rng.normal(3, 0.8, 2) + [0, 0.6])) for i in range(5)],
            style=style,
            ylabel="time in centre (s)",
        )

    def ranked(ax):
        figures.ranked_bars_with_floor(
            ax,
            ["speed", "turning", "rearing", "grooming", "freezing"],
            [0.92, 0.61, 0.44, 0.21, 0.12],
            0.5,
            style=style,
            xlabel="|effect size|",
        )

    def steps(ax):
        recall = np.linspace(0, 1, 40)
        figures.overlaid_step(
            ax,
            [
                ("soft kNN", list(recall), list(1 - recall**2.2)),
                ("mean exemplar", list(recall), list(1 - recall**1.3)),
            ],
            style=style,
            xlabel="recall",
            ylabel="precision",
        )
        figures.reference_line(ax, 0.17, label="prevalence", style=style)

    def band(ax):
        epochs = np.arange(60)
        for index, (label, rate) in enumerate((("train", 0.08), ("validation", 0.06))):
            curve = 0.85 * (1 - np.exp(-rate * epochs)) + rng.normal(0, 0.01, 60)
            figures.line_with_band(
                ax,
                epochs,
                curve,
                lower=curve - 0.03,
                upper=curve + 0.03,
                label=label,
                index=index,
                style=style,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("mAP50")
        figures.place_legend(ax, style=style)

    def hist(ax):
        figures.histogram(
            ax,
            rng.lognormal(0.2, 0.6, 2000),
            bins=40,
            style=style,
            xlabel="speed (cm/s)",
        )

    def confusion(ax):
        grid = rng.dirichlet(np.ones(5) * 0.6, 5) * 0.5 + np.eye(5) * 0.5
        figures.heatmap(
            ax,
            grid,
            row_labels=labels,
            col_labels=labels,
            style=style,
            vmin=0,
            vmax=1,
            colorbar_label="share",
            annotate=True,
            fmt="{:.1f}",
        )

    def diverging(ax):
        figures.heatmap(
            ax,
            rng.normal(0, 1, (5, 8)),
            row_labels=labels,
            col_labels=[f"w{i}" for i in range(8)],
            style=style,
            diverging=True,
            colorbar_label="z vs saline",
        )

    def scatter(ax):
        cloud = rng.normal(0, 1, (600, 2))
        picked = [rng.normal(c, 0.35, (60, 2)) for c in ((1.4, 0.9), (-1.2, 1.1))]
        figures.scatter_groups(
            ax,
            [
                ("all bouts", list(cloud[:, 0]), list(cloud[:, 1])),
                ("cluster 3", list(picked[0][:, 0]), list(picked[0][:, 1])),
                ("cluster 7", list(picked[1][:, 0]), list(picked[1][:, 1])),
            ],
            style=style,
            xlabel="UMAP 1",
            ylabel="UMAP 2",
            neutral_first=True,
        )

    return [
        ("observation_dots", dots),
        ("group_distribution", violins),
        ("grouped_bars", grouped_bars),
        ("group_curves", curves),
        ("paired_lines", paired),
        ("ranked_bars_with_floor", ranked),
        ("overlaid_step", steps),
        ("line_with_band", band),
        ("histogram", hist),
        ("heatmap", confusion),
        ("heatmap_diverging", diverging),
        ("scatter_groups", scatter),
    ]


def _primitives(style: FigureStyle, out: pathlib.Path) -> pathlib.Path:
    """Each chart form as its own single-column figure — the size it ships at — tiled to a sheet.

    One figure per form rather than one grid of axes, because constrained layout shares margins
    along a grid column: a legend in one panel pushed a heatmap's colorbar a column-width away
    from its map, which is a property of the contact sheet and not of the chart.
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    tiles = []
    work = out / f"_tiles_{style.theme}"
    work.mkdir(exist_ok=True)
    for name, draw in _primitive_panels(style):
        with figures.styled(style):
            fig, ax = plt.subplots(figsize=(style.single_col_width_in, 2.5))
            draw(ax)
            ax.set_title(name, loc="left", fontsize=style.type.annotation, pad=20)
        (path,) = figures.save_figure(fig, work / name, ("png",))
        tiles.append(mpimg.imread(path)[:, :, :3])
    return _tile(tiles, 3, style.chart.surface, out / f"primitives_{style.theme}.png")


def _tile(images, cols: int, background: str, path: pathlib.Path) -> pathlib.Path:
    """Lay ``images`` out ``cols`` wide on ``background``, each centred in an equal cell."""
    import matplotlib.image as mpimg

    from detail.core.color_checks import hex_to_rgb

    fill = np.array(hex_to_rgb(background), dtype=float) / 255
    height = max(i.shape[0] for i in images) + 24
    width = max(i.shape[1] for i in images) + 24
    rows = -(-len(images) // cols)
    sheet = np.ones((rows * height, cols * width, 3)) * fill
    for index, image in enumerate(images):
        r, c = divmod(index, cols)
        y = r * height + (height - image.shape[0]) // 2
        x = c * width + (width - image.shape[1]) // 2
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    mpimg.imsave(path, np.clip(sheet, 0, 1))
    return path


def _behavior_rows(rng):
    from detail.storage import behavior_output as bo

    names = ["rearing", "grooming", "walking", "still", "turning"]
    rows, coverage = [], []
    for r in range(6):
        recording = f"recording_{r + 1:03d}"
        length = 600.0 if r != 4 else 420.0
        coverage.append(
            bo.CoverageRow(recording, 0, 0, int(length * _NS), int(length * _NS), 18000)
        )
        t = 0.0
        while True:
            t += float(rng.exponential(9.0))
            duration = float(rng.lognormal(0.6, 0.8))
            if t + duration > length:
                break
            name = names[int(rng.integers(0, len(names)))]
            rows.append(
                bo.BehaviorInstance(
                    recording,
                    0,
                    int(t * _NS),
                    int((t + duration) * _NS),
                    int(t * 30),
                    int((t + duration) * 30),
                    f"c{names.index(name):02d}",
                    name,
                )
            )
            t += duration
    return rows, coverage


def _behavior_exports(style: FigureStyle, out: pathlib.Path) -> list[pathlib.Path]:
    """The three behavior-export figures, exactly as an export writes them."""
    rows, coverage = _behavior_rows(np.random.default_rng(11))
    written = []
    for name, fig in (
        ("ethogram", behavior_figures.ethogram(rows, coverage, style=style)),
        ("time_by_behavior", behavior_figures.time_by_behavior(rows, style=style)),
        ("bout_durations", behavior_figures.bout_durations(rows, style=style)),
    ):
        written += figures.save_figure(fig, out / f"{name}_{style.theme}", ("png",))
    return written


def _screen(theme_name: str, out: pathlib.Path) -> pathlib.Path | None:
    """Live GUI plots under the application's own palette in ``theme_name``, grabbed as PNG.

    Built through ``gui/plot_theme.py`` the way a page builds them, inside a window drawn with
    Fusion's real palette for that scheme — so what is judged is a plot *on its page*, which is
    where the old white panels failed. ``None`` when Qt cannot start (no display, no PyQt6).
    """
    try:
        from detail.gui.qt import QApplication, QGridLayout, QLabel, QWidget, Qt
    except ImportError:
        return None
    import pyqtgraph as pg

    from detail.gui import plot_theme

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    scheme = Qt.ColorScheme.Dark if theme_name == "dark" else Qt.ColorScheme.Light
    app.styleHints().setColorScheme(scheme)
    app.processEvents()
    theme = plot_theme.current()
    marks = plot_theme.MARKS
    rng = np.random.default_rng(3)

    page = QWidget()
    grid = QGridLayout(page)
    grid.addWidget(QLabel(f"GUI plots — application in {theme_name} mode"), 0, 0, 1, 2)

    curve = plot_theme.themed_plot()
    curve.setLabel("bottom", "epoch")
    curve.setLabel("left", "mAP50")
    plot_theme.add_legend(curve, corner="bottom-right")
    epochs = np.arange(60)
    for index, (label, rate) in enumerate((("train", 0.08), ("validation", 0.06))):
        y = 0.85 * (1 - np.exp(-rate * epochs)) + rng.normal(0, 0.01, 60)
        low = pg.PlotDataItem(epochs, y - 0.03)
        high = pg.PlotDataItem(epochs, y + 0.03)
        curve.addItem(
            pg.FillBetweenItem(
                low,
                high,
                brush=plot_theme.brush(theme.identity(index), plot_theme.fill_alpha()),
            )
        )
        curve.plot(epochs, y, pen=plot_theme.pen(theme.identity(index)), name=label)
    grid.addWidget(curve, 1, 0)

    strip = plot_theme.themed_plot()
    strip.setLabel("bottom", "speed (cm/s)")
    strip.hideAxis("left")
    background = rng.gamma(2, 3, 3000)
    counts, edges = np.histogram(background, bins=48, range=(0, 25))
    strip.addItem(
        pg.BarGraphItem(
            x=0.5 * (edges[:-1] + edges[1:]),
            height=counts / counts.max(),
            width=(edges[1] - edges[0]) * 0.92,
            brush=plot_theme.brush(theme.neutral),
            pen=None,
        )
    )
    members = rng.gamma(5, 2.2, 60)
    strip.addItem(
        pg.ScatterPlotItem(
            x=members,
            y=0.25 + 0.6 * rng.random(60),
            size=marks.marker_size,
            brush=plot_theme.brush(theme.identity(0)),
            pen=plot_theme.pen(theme.surface, width=marks.marker_ring),
        )
    )
    grid.addWidget(strip, 1, 1)

    bars = plot_theme.themed_plot()
    bars.setLabel("bottom", "confidence cutoff")
    bars.setLabel("left", "instances kept")
    xs = np.linspace(0, 1, 40)
    bars.plot(xs, 900 * (1 - xs) ** 2, pen=plot_theme.pen(theme.identity(0)))
    bars.addItem(
        pg.InfiniteLine(
            pos=0.62,
            angle=90,
            pen=plot_theme.pen(
                theme.reference, width=marks.reference_width, dashed=True
            ),
        )
    )
    grid.addWidget(bars, 2, 0)

    scatter = plot_theme.themed_plot()
    scatter.setLabel("bottom", "PC1")
    scatter.setLabel("left", "PC2")
    plot_theme.add_legend(scatter, corner="top-right")
    for index, centre in enumerate(((-1.0, 0.4), (1.2, -0.3), (0.2, 1.4))):
        points = rng.normal(centre, 0.45, (80, 2))
        scatter.addItem(
            pg.ScatterPlotItem(
                x=points[:, 0],
                y=points[:, 1],
                size=marks.marker_size,
                brush=plot_theme.brush(theme.identity(index)),
                pen=plot_theme.pen(theme.surface, width=marks.marker_ring),
                name=f"track {index}",
            )
        )
    grid.addWidget(scatter, 2, 1)

    page.resize(980, 620)
    page.show()
    for _ in range(3):
        app.processEvents()
    path = out / f"screen_{theme_name}.png"
    page.grab().save(str(path))
    page.close()
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "chart_gallery")
    parser.add_argument("--theme", choices=("light", "dark", "both"), default="both")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    themes = ("light", "dark") if args.theme == "both" else (args.theme,)
    for name in themes:
        style = FigureStyle(theme=name)
        for path in (
            _swatches(style, args.out),
            _primitives(style, args.out),
            _composed(style, args.out),
            *_behavior_exports(style, args.out),
        ):
            print(path)
        screen = _screen(name, args.out)
        print(screen if screen else f"(no Qt: skipped the {name} GUI plots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
