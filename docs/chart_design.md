# Chart design

**Agent-facing. Not a user page** — it is in `mkdocs.yml`'s `exclude_docs`.

Every chart this package draws — an exported figure, a live plot in the GUI, a hand-drawn lane —
takes its colours, sizes and weights from one module. This is the reasoning behind that module:
the measurements that picked the palette, the rules the chart forms follow, and how to change
either without breaking the guards. **The values themselves live in code and nowhere else**:
`detail/core/chart_theme.py`. A number restated here is a second copy, so this page names the
decisions and points at the code.

## One design, three adapters

| Drawn with | Adapter | Used for |
| --- | --- | --- |
| matplotlib | `detail/report/style.py` (rcParams, colormaps) + `report/figures.py` (the primitives) | every exported figure; the Comparison page shows the PNGs |
| pyqtgraph | `detail/gui/plot_theme.py` (`themed_plot`, `pen`, `brush`) | every live plot in the GUI |
| QPainter | `plot_theme.current()` + `QColor(theme.<role>)` | the lanes and maps: event / span / count / track strips, the ethogram strip, the spatial map |

`core/chart_theme.py` is pure data — no Qt, no matplotlib, no numpy — because all three import it
(`tests/core/test_chart_theme.py` scans its imports). `core/visualize.PALETTE_OKABE_ITO` is the
same identities as RGB, in the same order, so a track or a cluster is one colour on a video frame
and in the chart beside it.

**Before this module** the three paths each had their own idea of a chart: `setBackground("w")`
at eleven sites, `#444` for a median in one file and `(55, 62, 72)` in another, and two copies of
Okabe-Ito **in different orders** — the first cluster was orange on screen and blue in its
exported ethogram. In a dark window every plot was a lit panel with grey tick labels.

## Colour does one job at a time

| Role | Job | Rule |
| --- | --- | --- |
| `categorical` / `identity(i)` | which group, which behavior | fixed order; the first six are validated, then a frozen extension |
| `hue(name)` | a category whose colour *means* something (TP green, FP vermillion) | ask by name — never index the palette for a meaning |
| `neutral` | everything that is not the subject: the run's spread, unselected clusters, Other | quieter than any series, but a mark |
| `summary` | a median, a mean, an error bar | ink, never the group's colour — it would vanish into the group's points |
| `reference` | chance, a null floor, a threshold | dashed, labelled in muted text outside the plot area |
| `handle` | a control drawn on a plot — the cutoff you drag | one colour on every plot that carries one |
| `text` / `text_secondary` / `text_muted` | titles / ticks and legends / annotations | text never wears a series colour |
| `sequential` | magnitude | no hue, lightness alone; the low end sits near the surface so zero reads as nothing |
| `diverging` | signed difference | blue ↔ grey ↔ yellow, centred on zero; the poles differ in lightness too, and no behavior label takes a colour near either (`ChartTheme.label_identity`) |

Status (ready / warning / error) is not a chart colour and stays in `gui/status_color.py`.

## The palette, and how it was chosen

Okabe-Ito, the palette reviewers in the field recognise, **minus black** (it is the ink, so a
series in it reads as an axis) **and yellow** (1.3:1 on white; stepped into the lightness band it
turns olive). The six are ordered by search, not taste: every ordering was run through the checks
in `core/color_checks.py` in both themes, and the first order passing all of them with blue and
orange leading was taken. The order it replaced measured a worst neighbouring pair of ΔE 7.6
under deuteranopia against a target of 8; `test_the_cvd_gate_would_catch_the_order_it_replaced`
pins that.

**The dark theme is the same six stepped, not an inversion.** Only blue moves: `#0072B2` is 2.9:1
on the dark surface, and lifting it is bounded above by sky blue, which is the same hue separated
only by lightness. An early search that optimised separation instead put the two on one colour
while every adjacent-pair check passed — which is why `color_checks.nearest_pair` exists, a gate
over *all* pairs that the reference method does not have.

**The dark lightness band is wider than the reference method's** (`LIGHTNESS_BAND`): its 0.67
ceiling is calibrated for a palette with no same-hue pair, and applied here it turns amber and sky
blue olive and ochre and drops sky blue onto blue.

**Two light slots sit below 3:1 on white** — amber and sky blue, canonical Okabe-Ito values. The
relief rule applies: an exported figure ships its CSV, and a GUI plot sits beside its table. The
test pins exactly those two, so a *new* low-contrast colour fails.

**Past six identities** (a clustering names thirty behaviors) the frozen `extended` list takes
over: generated once by greedy max-min distance from the six and the neutrals, worst pair ΔE 9.3.
Distinct at a glance, **not** colour-blind safe, so a chart that reaches it must name its marks
another way too. The first six never move when a seventh arrives; the HSV fallback this replaced
re-hued the whole vocabulary.

**Scatter caps at three.** Only the first three slots pass the all-pairs check a scatter needs;
past three the legend carries identity.

## Type, weights, sizes

Print sizes are journal-standard final sizes (`PRINT_TYPE`, `PRINT_MARKS`, the column widths):
Nature and Cell ask for 5-7 pt at printed size; this sits at the readable end with a 6 pt floor
nothing drops below. The face is the first of Arial → Helvetica → Liberation Sans → DejaVu Sans
this machine has (`report/style.installed_families`) — filtered, because matplotlib logs a warning
per missing family on every figure otherwise. Screen sizes (`SCREEN_TYPE`, `SCREEN_MARKS`) are
relative to the application font, in pixels.

Axes: left and bottom only, ticks outward, no grid unless a chart earns one. Markers carry a thin
ring in the surface colour so overlapping points stay separable; a legend key drops the ring, or
it reads as a dashed line.

## Rules the chart forms follow

- **Small groups are points.** Below `figures.VIOLIN_MIN_N` in any group, `group_distribution`
  draws every observation and a median instead of a violin — at three animals a group a density
  estimate is a shape invented from three numbers. The choice is made for the whole chart.
- **One series, one colour.** Ranked bars were once a hue per rank, spending the identity channel
  on nothing.
- **Legends go outside the plot area** in an export (right, top-aligned); the strip above the axes
  carries the caption. A GUI legend sits in a corner the site names (`add_legend(corner=…)`),
  because only the site knows where its data is not.
- **Colliding category labels are turned by as little as works**, measured at save time
  (`figures._turn_colliding_tick_labels`), never by a character-count rule.
- **Constrained layout on every figure**, and `save_figure` applies the save-time settings itself:
  behavior exports were saved after their style context closed and wrote Type-3 fonts, a fallback
  face and SVGs with random element ids.
- **Words that name a colour by lightness are wrong in one theme.** "This cluster's in dark"
  became "in colour"; a density is "taller" or "more colour", not "darker".

## Light and dark

A GUI plot follows the application: `plot_theme.current()` reads the colour scheme, or the
palette's lightness when the platform reports none (the offscreen test platform never does). A
themed plot re-themes its chrome when its palette changes; its marks follow on their next draw.
The plot surface is one step lighter than the window in both themes, so a plot is a card.

**Exports are light unless asked.** `theme="dark"` reaches every surface that writes a figure —
`Comparison.export(theme=…)`, `Experiment.run_export(..., theme=…)` for every target that draws,
`detail comparison export --theme`, `detail export run --theme`, and a *Dark background* box in
the export dialog and on the Comparison page. The comparison manifest records `figure_theme`.

## Changing the design

1. Edit `core/chart_theme.py`.
2. `pytest tests/core/test_chart_theme.py` — every colour through every check, in both themes.
3. `python scripts/chart_gallery.py --out <dir>` and **look at it**: swatches with the measured
   numbers, every primitive at its export size, the behavior exports, and the GUI plots on both
   surfaces. The checks measure colour; only the picture shows a collision.

The guards: `tests/core/test_chart_theme.py` (the palettes), `tests/report/test_figures.py` (ink,
size floor, collisions, save-time settings, in both themes), `tests/gui/test_plot_theme.py`
(a plot in both themes and across a switch), and `gui/requirements.py`'s
`a_plot_takes_its_colours_from_the_chart_theme` (no plot styled outside the adapter).
