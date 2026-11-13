"""The Cluster profile window: what sets one cluster apart, and how far to trust it (§22).

One window, opened from three places — the Interpretation tree's right-click menu, the clip
window's *Cluster details*, and the clip montage — so a fix to it reaches every one of them.
It shows a saved profile at once when there is one, and computes a new one off the UI thread
(``gui/busy.run_progress``), naming each phase.

**The verdict comes first and gates the chart.** A classifier always ranks *something*, so the
held-out score and its floor are the first line, and when the score does not clear the floor
by the margin the chart stays folded behind *Show anyway* — a chart that leads would be read as
an answer it is not.

Every setting is built from ``configs/cluster_profile.ClusterProfileConfig`` through
``gui/_pydantic_form``, so its range, default and help are the CLI's and the API's.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.analysis import explainers
from detail.api.cluster_profile import place_text
from detail.configs.cluster_profile import ClusterProfileConfig
from detail.gui import plot_theme
from detail.gui._pydantic_form import any_field_value, any_field_widget, field_meta
from detail.gui._pydantic_form import set_any_field_value
from detail.gui.busy import run_progress
from detail.gui.dialog import Dialog
from detail.gui.lifetime import discard
from detail.gui.plots import bound_axes
from detail.gui.status_color import MUTED, READY, WARN, set_label_color

#: The settings shown in the first row; the rest in the second.
_MAIN_FIELDS = ("compare", "against", "explainer", "time_budget_s")
_MORE_FIELDS = ("margin", "n_per_side", "leave_out_place", "seed")
#: What a choice reads as on screen, where its stored value is an id.
_CHOICE_WORDS = {
    "compare": {
        "rest": "Rest of the run",
        "sibling": "Its sibling",
        "node": "Another cluster",
    }
}
#: The measures drawn on the chart, strongest first.
_CHART_ROWS = 8


class ClusterProfileDialog(Dialog):
    """The profile of ``node`` of clustering ``run``. ``title`` names the node as its opener
    does ("Cluster #1394 (rearing)")."""

    def __init__(self, experiment, run, node: int, *, title: str, parent=None):
        super().__init__(parent)
        self._experiment = experiment
        self._run = run
        self._node = int(node)
        self.setWindowTitle(f"{title} · cluster profile")
        outer = self.body()

        settings = QGridLayout()
        self._fields: dict[str, QWidget] = {}
        for row, names in enumerate((_MAIN_FIELDS, _MORE_FIELDS)):
            for col, name in enumerate(names):
                widget = any_field_widget(ClusterProfileConfig, name)
                # Display words only; the item data stays the model's value.
                words = _CHOICE_WORDS.get(name, {})
                if name == "explainer":
                    words = {k: v.label for k, v in explainers.EXPLAINERS.items()}
                for i in range(widget.count() if words else 0):
                    if widget.itemData(i) in words:
                        widget.setItemText(i, words[widget.itemData(i)])
                self._fields[name] = widget
                settings.addWidget(
                    QLabel(f"{field_meta(ClusterProfileConfig, name)['label']}:"),
                    row,
                    col * 2,
                )
                settings.addWidget(widget, row, col * 2 + 1)
        outer.addLayout(settings)
        # Unset reads as "none chosen", not the form's "auto": nothing is worked out for it.
        self._fields["against"].setSpecialValueText("—")
        self._fields["compare"].currentIndexChanged.connect(
            lambda _i: self._sync_against()
        )
        self._sync_against()

        action_row = QHBoxLayout()
        self._compute = QPushButton("Profile this cluster")
        self._compute.clicked.connect(self._recompute)
        action_row.addWidget(self._compute)
        self._status = QLabel("")
        set_label_color(self._status, MUTED)
        action_row.addWidget(self._status, 1)
        outer.addLayout(action_row)

        self._result = QVBoxLayout()
        outer.addLayout(self._result)
        self._result_widgets: list[QWidget] = []
        outer.addStretch()

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

        saved = experiment.cluster_profile(str(run.id), self._node)
        if saved is None:
            self._place(
                QLabel(
                    "No profile of this cluster yet. Profile it to see which measures set "
                    "it apart, checked on recordings the model has not seen."
                )
            )
        else:
            self._load_settings(saved.config)
            self.show_profile(saved)

    # -- settings ------------------------------------------------------------ #
    def config(self) -> dict:
        names = (*_MAIN_FIELDS, *_MORE_FIELDS)
        return {
            name: any_field_value(ClusterProfileConfig, name, self._fields[name])
            for name in names
        }

    def _load_settings(self, config: dict) -> None:
        for name, widget in self._fields.items():
            if name in config:
                set_any_field_value(ClusterProfileConfig, name, widget, config[name])
        self._sync_against()

    def _sync_against(self) -> None:
        """The other cluster is asked for only when comparing against one, and kept unset
        otherwise, as the config requires."""
        against = self._fields["against"]
        on = self._fields["compare"].currentData() == "node"
        against.setEnabled(on)
        if not on:
            set_any_field_value(ClusterProfileConfig, "against", against, None)

    # -- compute ------------------------------------------------------------- #
    def _recompute(self) -> None:
        experiment, run_id, node, config = (
            self._experiment,
            str(self._run.id),
            self._node,
            self.config(),
        )
        run_progress(
            self,
            lambda emit: experiment.profile_cluster(  # off-thread (run_progress)
                run_id, node, config, progress=emit
            ),
            self.show_profile,
            label="Profiling this cluster…",
            on_error=self._failed,
            supersedes="cluster profile",
        )

    def _failed(self, exc) -> None:
        self._status.setText(f"Could not profile: {exc}")
        set_label_color(self._status, WARN)

    # -- the way to SHAP, where the profile fell back without it -------------- #
    def _setup_row(self) -> QWidget:
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Install SHAP…")
        button.setToolTip(
            "Configuration ▸ Cluster profile: install SHAP into this app's "
            "environment, then Recompute"
        )
        button.clicked.connect(self._open_setup)
        box.addWidget(button)
        box.addStretch(1)
        return row

    def _open_setup(self) -> None:
        from detail.gui.profile_describe_setup import ProfileDescribeSetupDialog

        ProfileDescribeSetupDialog(parent=self).exec()

    # -- showing a profile --------------------------------------------------- #
    def _place(self, widget: QWidget) -> None:
        self._result.addWidget(widget)
        self._result_widgets.append(widget)

    def _clear(self) -> None:
        for widget in self._result_widgets:
            discard(widget)
        self._result_widgets = []

    def show_profile(self, profile) -> None:
        self._clear()
        self._profile = profile
        self._compute.setText("Recompute")
        self._status.setText(
            f"Made {profile.created_at.replace('T', ' ').rstrip('Z')} UTC · took "
            f"{profile.elapsed_s:.0f} s · {profile.n_units_read} animal-recordings read"
        )
        set_label_color(self._status, MUTED)
        e = profile.explanation

        verdict = QLabel(profile.headline)
        verdict.setWordWrap(True)
        font = verdict.font()
        font.setBold(True)
        verdict.setFont(font)
        set_label_color(verdict, READY if e.separated else WARN)
        self._place(verdict)
        # Against another cluster: whether position and facing alone tell the two apart.
        place_line = place_text(profile)
        if place_line:
            place = QLabel(place_line)
            place.setWordWrap(True)
            self._place(place)

        compared = {
            "rest": "the rest of the run",
            "sibling": f"its sibling #{profile.comparison_node}",
            "node": f"cluster #{profile.comparison_node}",
        }.get(profile.compare, profile.compare)
        basis = QLabel(
            f"Based on: {profile.basis} · {len(e.measures_used)} measures · "
            f"{e.n_cluster} members against {e.n_comparison} from {compared}"
            f" · {explainers.EXPLAINERS[e.explainer].label}"
        )
        basis.setWordWrap(True)
        set_label_color(basis, MUTED)
        self._place(basis)
        for line in _notes(profile):
            note = QLabel(line)
            note.setWordWrap(True)
            set_label_color(note, MUTED)
            self._place(note)
        if fell_back(profile):
            self._place(self._setup_row())

        if e.importances:
            apart = QGroupBox("What sets it apart")
            column = QVBoxLayout(apart)
            chart = _importance_chart(profile)
            column.addWidget(chart)
            legend = QLabel(
                "Bars to the right push toward this cluster, to the left away from it. "
                "Beside each name: this cluster's median against the comparison's."
            )
            legend.setWordWrap(True)
            set_label_color(legend, MUTED)
            column.addWidget(legend)
            if not e.separated:
                # Folded, not removed: the numbers are still worth a look, but not first.
                reveal = QCheckBox("Show anyway (the measures do not separate it)")
                chart.setVisible(False)
                legend.setVisible(False)
                reveal.toggled.connect(chart.setVisible)
                reveal.toggled.connect(legend.setVisible)
                column.insertWidget(0, reveal)
            self._place(apart)

        if e.rules:
            rules = QGroupBox("In plain rules (a 3-level decision tree)")
            column = QVBoxLayout(rules)
            for rule in e.rules:
                line = QLabel(
                    f"• {rule.text} → this cluster (right {rule.precision:.0%} of the "
                    f"time, covers {rule.coverage:.0%} of its members)"
                )
                line.setWordWrap(True)
                column.addWidget(line)
            self._place(rules)

        context = _context_box(profile)
        if context is not None:
            self._place(context)


def fell_back(profile) -> bool:
    """The profile asked for SHAP and got permutation importance, because shap is missing."""
    e = profile.explanation
    return e.explainer == explainers.FALLBACK_EXPLAINER and bool(e.note)


def _notes(profile) -> list[str]:
    e = profile.explanation
    out = []
    if e.note:
        out.append(f"Note: {e.note}.")
    from detail.api.cluster_profile import LEFT_OUT_PLACE

    place = [n for n, why in e.measures_dropped.items() if why == LEFT_OUT_PLACE]
    others = {n: why for n, why in e.measures_dropped.items() if n not in place}
    if others:
        skipped = "; ".join(
            f"{profile.labels.get(name, name)} ({why})" for name, why in others.items()
        )
        out.append(f"Skipped: {skipped}.")
    if place:
        out.append(
            "Left out, as set: "
            + ", ".join(profile.labels.get(n, n) for n in place)
            + " (where the animal is and which way it faces)."
        )
    if e.rows_dropped:
        out.append(
            f"{e.rows_dropped} members left out for a gap in one of the measures."
        )
    return out


def _display_name(profile, name: str) -> str:
    unit = profile.units.get(name, "")
    label = profile.labels.get(name, name)
    return f"{label} ({unit})" if unit else label


def _importance_chart(profile) -> QWidget:
    """Horizontal bars, strongest at the top: length is the weight, side is the direction."""
    rows = profile.explanation.importances[:_CHART_ROWS]
    theme = plot_theme.current()
    toward, away = theme.diverging[-1], theme.diverging[0]
    plot = plot_theme.themed_plot()
    plot.setMinimumHeight(40 + 34 * len(rows))
    plot.setMouseEnabled(x=False, y=False)
    plot.hideButtons()
    ys = np.arange(len(rows))[::-1].astype(float)
    signed = [i.value * (i.direction or 1) for i in rows]
    for y, value, item in zip(ys, signed, rows):
        color = toward if item.direction >= 0 else away
        plot.addItem(
            pg.BarGraphItem(
                x0=[0.0],
                y=[y],
                width=[value],
                height=0.6,
                brush=plot_theme.brush(color),
                pen=None,
            )
        )
    plot.getAxis("left").setTicks(
        [
            [
                (
                    float(y),
                    f"{_display_name(profile, item.name)}   "
                    f"{item.cluster_median:.3g} vs {item.comparison_median:.3g}",
                )
                for y, item in zip(ys, rows)
            ]
        ]
    )
    plot.addLine(x=0, pen=plot_theme.pen(theme.axis))
    span = max([abs(v) for v in signed] + [1e-9]) * 1.15
    plot.setXRange(-span, span, padding=0)
    plot.setYRange(-0.6, len(rows) - 0.4, padding=0)
    # A weight is signed and bounded by the strongest bar; the rows are what there is.
    bound_axes(plot, x=(-span, span), y=(-0.6, len(rows) - 0.4))
    plot.setLabel("bottom", "weight in the fit (toward → / ← away)")
    return plot


def _context_box(profile) -> QGroupBox | None:
    c = profile.context
    rows: list[tuple[str, str]] = []

    def neighbours(items) -> str:
        return ", ".join(
            f"{n['name'] or '#' + str(n['node'])} ({n['share']:.0%})" for n in items
        )

    if c.before:
        rows.append(("Usually comes after", neighbours(c.before)))
    if c.after:
        rows.append(("Usually leads to", neighbours(c.after)))
    if np.isfinite(c.bout_median_s):
        rows.append(
            (
                "Typical length",
                f"{c.bout_median_s:.2f} s median ({c.bout_p10_s:.2f} – "
                f"{c.bout_p90_s:.2f} s)",
            )
        )
    if c.session_thirds:
        early, middle, late = c.session_thirds
        rows.append(
            (
                "When in the session",
                f"{early:.0%} early · {middle:.0%} middle · {late:.0%} late",
            )
        )
    for o in c.over_represented:
        rows.append(
            (
                "Over-represented in",
                f"{o['column']} = {o['value']}: {o['ratio']:.2f}× the run's share "
                f"(chance reaches {o['floor']:.2f}×)",
            )
        )
    if not rows:
        return None
    box = QGroupBox("Context")
    grid = QGridLayout(box)
    for i, (key, value) in enumerate(rows):
        name = QLabel(key)
        font = name.font()
        font.setBold(True)
        name.setFont(font)
        grid.addWidget(name, i, 0)
        text = QLabel(value)
        text.setWordWrap(True)
        grid.addWidget(text, i, 1)
    grid.setColumnStretch(1, 1)
    return box
