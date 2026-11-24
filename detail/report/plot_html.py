"""Draw a ``report/plot_spec.PlotSpec`` as a self-contained interactive HTML page (plotly).

The matplotlib renderer's twin (``report/plot_render.py``): the same spec, the same colour roles
resolved against the chart theme, the same panels, axes, colour scale, legend and note — but
hoverable, zoomable and, for 3-D, turnable. ``plotly.js`` is embedded, so the page opens offline
with no server. Every 3-D panel shares one camera, as the static grid does: turning one turns
them all.

``plotly`` is imported inside :func:`figure`, never at module import (it is a dependency, but the
package must import without it the way it imports without a backend).
"""

import json
import math
from pathlib import Path

import numpy as np

from detail.core import chart_theme
from detail.report.plot_spec import Line, PlotSpec
from detail.report.style import plot_stops

#: Pixels per panel; the page is this times its grid.
PANEL_PX = 420

#: Most panels a row holds (``plot_render.MAX_COLUMNS``).
MAX_COLUMNS = 10


def _columns(n: int) -> int:
    return max(1, min(MAX_COLUMNS, math.ceil(math.sqrt(n))))


def _scale(colors: tuple[str, ...]) -> list[list]:
    steps = len(colors) - 1
    return [[i / steps if steps else 0.0, c] for i, c in enumerate(colors)]


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = chart_theme.rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def _finite_list(values) -> list:
    """A JSON-safe list: NaN becomes None, which plotly draws as a gap."""
    arr = np.asarray(values, dtype=np.float64)
    return [None if not np.isfinite(v) else float(v) for v in arr]


def figure(spec: PlotSpec, theme: str = "light"):
    """The plotly ``Figure`` for ``spec`` on the ``theme`` surface."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    chart = chart_theme.theme(theme)
    n = len(spec.panels)
    cols = _columns(n)
    rows = -(-n // cols)
    kind = "scene" if spec.three_d else "xy"
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": kind} for _c in range(cols)] for _r in range(rows)],
        subplot_titles=[p.title for p in spec.panels] + [""] * (rows * cols - n),
        horizontal_spacing=0.06 if not spec.three_d else 0.02,
        vertical_spacing=min(0.12, 0.6 / rows) if rows > 1 else 0.1,
    )
    has_heat = any(p.heat is not None for p in spec.panels)
    # The picture's own colours (``report/style.plot_stops``), so the page and the PNG agree.
    ramp = plot_stops(chart, heat=has_heat)
    if spec.color is not None:
        fig.update_layout(
            coloraxis={
                "colorscale": _scale(tuple(ramp)),
                "cmin": spec.color.vmin,
                "cmax": spec.color.vmax,
                "colorbar": {"title": {"text": spec.color.label}},
            }
        )
    legend_seen: set[tuple] = set()
    for index, panel in enumerate(spec.panels):
        row, col = index // cols + 1, index % cols + 1
        if panel.heat is not None:
            heat = panel.heat
            fig.add_trace(
                go.Heatmap(
                    # A cell the animal never entered is left empty, as the PNG leaves it.
                    z=[
                        [v if v > 0 else None for v in row]
                        for row in heat.grid.tolist()
                    ],
                    x=((heat.x_edges[:-1] + heat.x_edges[1:]) / 2).tolist(),
                    y=((heat.y_edges[:-1] + heat.y_edges[1:]) / 2).tolist(),
                    coloraxis="coloraxis",
                    hovertemplate="x %{x:.3g}<br>y %{y:.3g}<br>%{z:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        for band in panel.bands:
            colour = chart.identity(band.identity)
            shape = "hvh" if band.step else "linear"
            fig.add_trace(
                go.Scatter(
                    x=_finite_list(band.x),
                    y=_finite_list(band.hi),
                    mode="lines",
                    line={"width": 0, "shape": shape},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=_finite_list(band.x),
                    y=_finite_list(band.lo),
                    mode="lines",
                    line={"width": 0, "shape": shape},
                    fill="tonexty",
                    fillcolor=_rgba(colour, 0.25),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        for line in panel.lines:
            fig.add_trace(
                _line_trace(go, line, spec, chart, legend_seen), row=row, col=col
            )
        if panel.note:
            fig.add_annotation(
                text=panel.note,
                xref=(
                    f"x{index + 1 if index else ''} domain"
                    if not spec.three_d
                    else "paper"
                ),
                yref=(
                    f"y{index + 1 if index else ''} domain"
                    if not spec.three_d
                    else "paper"
                ),
                x=0,
                y=1.08,
                showarrow=False,
                font={"color": chart.text_muted, "size": 11},
                xanchor="left",
            )
    _dress_axes(fig, spec, n)
    title = spec.title + (f"<br><sup>{spec.note}</sup>" if spec.note else "")
    fig.update_layout(
        title={"text": title, "x": 0.01},
        paper_bgcolor=chart.surface,
        plot_bgcolor=chart.surface,
        font={"color": chart.text},
        width=max(PANEL_PX * cols + 160, 640),
        height=max(PANEL_PX * rows + 120, 420),
        showlegend=spec.legend,
        hovermode="closest",
        margin={"l": 60, "r": 40, "t": 90, "b": 50},
    )
    return fig


def _line_trace(go, line: Line, spec: PlotSpec, chart, legend_seen: set):
    x, y = _finite_list(line.x), _finite_list(line.y)
    if line.values is not None:
        # plotly refuses a missing colour; a gap's coordinates are None, so the point it
        # colours is never drawn and any finite stand-in will do.
        values = np.nan_to_num(
            np.asarray(line.values, dtype=np.float64), nan=0.0
        ).tolist()
        if line.z is not None:
            # A 3-D line takes a colour per vertex; the shared colour axis gives one scale.
            return go.Scatter3d(
                x=x,
                y=y,
                z=_finite_list(line.z),
                mode="lines",
                line={"color": values, "coloraxis": "coloraxis", "width": 3},
                name=line.label,
                showlegend=False,
                hovertemplate="x %{x:.3g}<br>y %{y:.3g}<br>t %{z:.3g} s<extra></extra>",
            )
        # A 2-D plotly line has one colour, so the path's colour rides on its markers over
        # a hairline of the path itself.
        return go.Scattergl(
            x=x,
            y=y,
            mode="lines+markers",
            line={"color": chart.neutral, "width": 1},
            marker={"color": values, "coloraxis": "coloraxis", "size": 3},
            name=line.label,
            showlegend=False,
            hovertemplate="x %{x:.3g}<br>y %{y:.3g}<extra>" + line.label + "</extra>",
        )
    colour = chart.identity(line.identity) if line.identity is not None else chart.text
    key = (line.identity, line.label)
    show = bool(line.label) and line.identity is not None and key not in legend_seen
    legend_seen.add(key)
    common = {
        "name": line.label,
        "legendgroup": line.label,
        "showlegend": show,
    }
    if line.z is not None:
        return go.Scatter3d(
            x=x,
            y=y,
            z=_finite_list(line.z),
            mode="lines",
            line={"color": colour, "width": 3},
            **common,
        )
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={"color": colour, "width": 2, "shape": "hvh" if line.step else "linear"},
        **common,
    )


def _dress_axes(fig, spec: PlotSpec, n: int) -> None:
    for index in range(n):
        suffix = "" if index == 0 else str(index + 1)
        if spec.three_d:
            y_range = spec.y.range
            if y_range and spec.invert_y:
                y_range = tuple(reversed(y_range))
            scene = {
                "xaxis": _axis(spec.x.label, spec.x.range),
                "yaxis": _axis(spec.y.label, y_range),
                "zaxis": _axis(spec.z.label, spec.z.range) if spec.z else {},
                "aspectmode": "cube",
            }
            if spec.camera is not None:
                scene["camera"] = _camera(spec.camera.elev, spec.camera.azim)
            fig.update_layout({f"scene{suffix}": scene})
            continue
        xaxis = {"title": spec.x.label}
        if spec.x.range:
            xaxis["range"] = list(spec.x.range)
        yaxis = {"title": spec.y.label}
        if spec.y.range:
            yaxis["range"] = (
                list(reversed(spec.y.range)) if spec.invert_y else list(spec.y.range)
            )
        if spec.aspect is not None:
            yaxis["scaleanchor"] = f"x{suffix}"
            # plotly's ratio is a y unit's pixels over an x unit's: the spec's aspect itself.
            yaxis["scaleratio"] = float(spec.aspect)
        fig.update_layout({f"xaxis{suffix}": xaxis, f"yaxis{suffix}": yaxis})


def _axis(title: str, value_range) -> dict:
    """A plotly axis: its title, and its range only when the spec gives one."""
    axis = {"title": title}
    if value_range:
        axis["range"] = list(value_range)
    return axis


def _camera(elev: float, azim: float) -> dict:
    """matplotlib's ``(elev, azim)`` as a plotly camera eye at the same angles."""
    r = 2.0
    e, a = math.radians(elev), math.radians(azim)
    return {
        "eye": {
            "x": r * math.cos(e) * math.cos(a),
            "y": r * math.cos(e) * math.sin(a),
            "z": r * math.sin(e),
        }
    }


#: Copies one scene's camera to every other when a person turns it, so the 3-D grid stays one
#: view. plotly links axes, not cameras, so this is the page's one script of its own.
#: Marks the script in the page, so a reader (or a test) can tell it from plotly.js itself.
LINK_CAMERAS_MARK = "detail: one camera for every 3-D panel"

_LINK_CAMERAS = """
// detail: one camera for every 3-D panel
var gd = document.getElementById('{plot_id}');
var syncing = false;
gd.on('plotly_relayout', function (ev) {
  if (syncing) { return; }
  var key = Object.keys(ev).find(function (k) { return /^scene\\d*\\.camera$/.test(k); });
  if (!key) { return; }
  var update = {};
  Object.keys(gd.layout).filter(function (k) { return /^scene\\d*$/.test(k); })
    .forEach(function (s) { update[s + '.camera'] = ev[key]; });
  syncing = true;
  Plotly.relayout(gd, update).then(function () { syncing = false; });
});
"""


def render_html(spec: PlotSpec, path: Path | str, *, theme: str = "light") -> Path:
    """Write ``spec`` to ``path`` as one self-contained HTML page (plotly.js embedded)."""
    import plotly.io as pio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = figure(spec, theme)
    page = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=True,
        div_id="location_plot",
        post_script=(
            _LINK_CAMERAS.replace("{plot_id}", "location_plot")
            if spec.three_d
            else None
        ),
        config={"displaylogo": False, "responsive": True},
    )
    page = page.replace(
        "<head>", f"<head><title>{json.dumps(spec.title)[1:-1]}</title>", 1
    )
    path.write_text(page, encoding="utf-8")
    return path
