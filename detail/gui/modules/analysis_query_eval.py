"""Behavior-query eval analysis module UI (§16).

Launch form: the embed run common to the selected (search) recordings, the search track,
the scored behaviors (the annotations' *covered* vocabulary), and the match / exemplar /
cross-validation knobs — the scalar params are generated from the config model
(``_pydantic_form``, the single source for range/help/label), so a widget can't drift
from the CLI or validator. Result view: the aggregate **mean ± std** metrics per behavior,
a per-fold breakdown, and the per-fold PR curves (read through ``QueryEvalPreview``,
Tenet 2), plus an **Explore clips…** button that opens a resizable window to scrub the
recording, review the GT/TP/FP/FN timeline, and **play** each clip (source over the pose
mask, the reused clip player).
"""

from typing import get_args

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from detail.analysis.behavior_labels import ground_truth_streams
from detail.configs.query_eval import _PERFRAME_METHODS, QueryEvalConfig
from detail.core.utils import coerce_series_key, subject_key
from detail.gui import frame_source
from detail.gui.collapse import CollapsibleBand
from detail.gui.dialog import Dialog
from detail.gui.forms import ConfigForm
from detail.gui.grip_splitter import GripSplitter
from detail.gui.help_ui import field_label
from detail.gui.modules._common import common_complete_run_items, repopulate_combo
from detail.gui._pydantic_form import (
    build_fields,
    field_meta,
    optional_spin,
    optional_value,
    set_optional_value,
)
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    register_analysis_module_ui,
)
from detail.gui import plot_theme
from detail.gui.plots import bound_axes
from detail.gui.tables import (
    BROWSE_ROWS,
    num_cell,
    enable_sorting,
    fill_width,
    fit_table_height,
    make_columns_resizable,
    row_id,
    stamp_row_id,
    top_of_pane,
)

_FUSIONS = list(get_args(QueryEvalConfig.model_fields["fusion"].annotation))
_METHODS = list(get_args(QueryEvalConfig.model_fields["method"].annotation))
_KEYFRAME_SELECTS = list(
    get_args(QueryEvalConfig.model_fields["keyframe_select"].annotation)
)
_NEGATIVE_POOLS = list(
    get_args(QueryEvalConfig.model_fields["negative_pool"].annotation)
)
# Scalar params rendered from the model, in form order.
_SCALARS = (
    "band_s",
    "k",
    "tau",
    "n_negatives",
    "n_keyframes",
    "iou_threshold",
    "point_window_s",
    "n_exemplars",
    "folds",
    "seed",
)
# Which scalar params each method actually uses — the rest are greyed out, so the form
# never implies a setting is in play when the chosen method ignores it.
_SOFT_KNN_FIELDS = {"k", "tau", "n_negatives"}
_KEYFRAME_FIELDS = {"n_keyframes"}


def _fields_for(method: str) -> set[str]:
    """The method-specific scalar fields ``method`` uses."""
    if method == "soft_knn":
        return _SOFT_KNN_FIELDS
    if method == "keyframe":
        return _KEYFRAME_FIELDS
    return set()


def _read_only_table(headers: list[str]) -> tuple[QTableView, QStandardItemModel]:
    model = QStandardItemModel()
    model.setHorizontalHeaderLabels(headers)
    table = QTableView()
    fill_width(table)
    table.setModel(model)
    table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    make_columns_resizable(table)
    enable_sorting(table)
    return table, model


class _QueryEvalConfig(ConfigWidget):
    def __init__(self) -> None:
        super().__init__()
        defaults = QueryEvalConfig(embed_run_id="")  # embed comes from the combo
        self._fields = build_fields(QueryEvalConfig)
        for name, (_, widget) in self._fields.items():
            widget.setValue(getattr(defaults, name))
        self._form = ConfigForm(self, QueryEvalConfig)
        form = self._form
        self._behavior_types: dict[str, str] = {}  # behavior -> "point" | "state"

        self._embed = QComboBox()
        form.addRow(
            field_label("query_eval", "embed_run_id", "Embed run:"), self._embed
        )
        self._add(form, "track_id")

        # Which ground-truth stream the behaviors come from. A recording carrying two
        # behavior streams would otherwise silently score against whichever is first.
        self._stream = QComboBox()
        self._stream.setToolTip(field_meta(QueryEvalConfig, "stream_name")["help"])
        self._stream.currentIndexChanged.connect(self._on_stream_selected)
        self._by_stream: dict[object, tuple[list[str], dict[str, str]]] = {}
        form.addRow(
            field_label("query_eval", "stream_name", "Ground-truth stream:"),
            self._stream,
        )

        self._behaviors = QListWidget()
        self._behaviors.setMaximumHeight(120)
        self._behaviors.itemChanged.connect(lambda _item: self._update_type_fields())
        form.addRow(
            field_label("query_eval", "behaviors", "Behaviors:"), self._behaviors
        )

        meta = field_meta(QueryEvalConfig, "fusion")
        self._fusion = QComboBox()
        self._fusion.addItems(_FUSIONS)
        self._fusion.setCurrentText(defaults.fusion)
        self._fusion.setToolTip(meta["help"])
        form.addRow(
            field_label("query_eval", "fusion", f"{meta['label']}:"), self._fusion
        )

        # Enums are not generated by _pydantic_form (it handles scalars), so the matcher
        # and its keyframe mode are hand-built — metadata still from the model.
        self._method = self._enum_combo(form, "method", _METHODS, defaults.method)
        self._method.currentTextChanged.connect(lambda _t: self._update_method_fields())

        for name in _SCALARS:
            self._add(form, name)

        self._keyframe_select = self._enum_combo(
            form, "keyframe_select", _KEYFRAME_SELECTS, defaults.keyframe_select
        )
        self._negative_pool = self._enum_combo(
            form, "negative_pool", _NEGATIVE_POOLS, defaults.negative_pool
        )

        # Optional (X | None) knobs the generator skips — built one at a time, with the
        # "omit" sentinel one step below the range the field allows.
        gap_meta = field_meta(QueryEvalConfig, "trajectory_split_gap_s")
        self._split_gap = optional_spin(
            QueryEvalConfig, "trajectory_split_gap_s", "(auto)"
        )
        form.addRow(
            field_label(
                "query_eval", "trajectory_split_gap_s", f"{gap_meta['label']}:"
            ),
            self._split_gap,
        )

        res_meta = field_meta(QueryEvalConfig, "max_results")
        self._max_results = optional_spin(QueryEvalConfig, "max_results", "(unbounded)")
        form.addRow(
            field_label("query_eval", "max_results", f"{res_meta['label']}:"),
            self._max_results,
        )

        hold_meta = field_meta(QueryEvalConfig, "n_holdout")
        self._n_holdout = optional_spin(QueryEvalConfig, "n_holdout", "(= exemplars)")
        form.addRow(
            field_label("query_eval", "n_holdout", f"{hold_meta['label']}:"),
            self._n_holdout,
        )

        self._update_type_fields()  # gate IoU / point-window on the checked behaviors
        self._update_method_fields()  # and the matcher knobs on the chosen matcher
        form.finish()

    def _add(self, form: ConfigForm, name: str) -> None:
        label, widget = self._fields[name]
        form.addRow(field_label("query_eval", name, f"{label}:"), widget)

    def _enum_combo(
        self, form: ConfigForm, name: str, choices: list[str], initial: str
    ) -> QComboBox:
        """A combo for a Literal field, labelled + explained from the config model."""
        meta = field_meta(QueryEvalConfig, name)
        combo = QComboBox()
        combo.addItems(choices)
        combo.setCurrentText(initial)
        combo.setToolTip(meta["help"])
        form.addRow(field_label("query_eval", name, f"{meta['label']}:"), combo)
        return combo

    # -- launch-dialog hooks ------------------------------------------------ #
    # Split resolve/apply (not the cheap on_recordings_changed): covered_behaviors reads the
    # canonical stream CSV for a *legacy* (pre-metadata) behavior stream, so the per-recording
    # reads run off the UI thread (CLAUDE.md — no O(recordings) disk I/O on the UI thread).
    def resolve_recordings(self, recordings: list) -> object:
        embed_items = common_complete_run_items(recordings, "embedding")
        # Resolve every behavior stream's vocabulary here, off the UI thread, so
        # switching the stream combo is a dict lookup rather than a fresh disk sweep.
        by_stream: dict[object, tuple[set[str], dict[str, str]]] = {}

        def add(key, covered, types):
            entry = by_stream.setdefault(key, (set(), {}))
            entry[0].update(covered)
            entry[1].update(types)

        for s in recordings:
            if not s.is_annotated:
                continue
            add(None, s.covered_behaviors(), s.behavior_types())
            for manifest in ground_truth_streams(s):
                name = manifest.get("name")
                add(name, s.covered_behaviors(name), s.behavior_types(name))
        return embed_items, {k: (sorted(c), t) for k, (c, t) in by_stream.items()}

    def apply_recordings(self, recordings: list, resolved: object) -> None:
        embed_items, by_stream = resolved
        repopulate_combo(self._embed, embed_items)
        self._by_stream = by_stream
        previous = self._stream.currentData()
        self._stream.blockSignals(True)
        self._stream.clear()
        self._stream.addItem("(first behavior stream)", None)
        for name in sorted(k for k in by_stream if k is not None):
            self._stream.addItem(name, name)
        index = self._stream.findData(previous)
        self._stream.setCurrentIndex(index if index >= 0 else 0)
        self._stream.blockSignals(False)
        # Same stream, new recordings: keep whatever the user had checked.
        self._refresh_behaviors()

    def _on_stream_selected(self, *_args) -> None:
        # A different stream is a different vocabulary, so the previous stream's checked
        # names carry no meaning — clearing first makes _set_behaviors check them all
        # (its first-populate rule), rather than leaving every box unchecked.
        self._behaviors.clear()
        self._refresh_behaviors()

    def _refresh_behaviors(self) -> None:
        covered, types = self._by_stream.get(self._stream.currentData(), ([], {}))
        self._behavior_types = types
        self._set_behaviors(covered)
        self._update_type_fields()

    def _set_behaviors(self, names: list[str]) -> None:
        checked = set(self._checked_behaviors())
        self._behaviors.blockSignals(True)  # repopulating is not a user edit
        self._behaviors.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Check known-checked names; on the first populate (nothing checked yet),
            # check all so a single-behavior experiment runs with no extra clicks.
            on = name in checked or not checked
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self._behaviors.addItem(item)
        self._behaviors.blockSignals(False)

    def _checked_behaviors(self) -> list[str]:
        return [
            self._behaviors.item(i).text()
            for i in range(self._behaviors.count())
            if self._behaviors.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _update_type_fields(self) -> None:
        """IoU applies only to **state** behaviors, the point window only to **point**
        behaviors — disable each control when no checked behavior is of that type (a
        state-only eval never uses the point window, and a point-only eval never uses IoU).
        The disabled field is greyed, not editable, and its value is ignored downstream.
        """
        types = {
            self._behavior_types.get(b, "state") for b in self._checked_behaviors()
        }
        self._set_type_field(
            "point_window_s",
            "point" in types,
            "Only used for point behaviors — none of the checked behaviors is a point.",
        )
        self._set_type_field(
            "iou_threshold",
            any(t != "point" for t in types),
            "Only used for state behaviors — none of the checked behaviors is a state.",
        )

    def _update_method_fields(self) -> None:
        """Grey out every method setting the chosen method does not use — including
        **band**, which the per-frame methods ignore (they score frames, not windows).
        """
        method = self._method.currentText()
        used = _fields_for(method)
        for name in _SOFT_KNN_FIELDS | _KEYFRAME_FIELDS:
            self._set_type_field(
                name,
                name in used,
                f"Only used by some methods — not by {method}.",
            )
        self._set_type_field(
            "band_s",
            method not in _PERFRAME_METHODS,
            "Not used by the per-frame methods — they score frames, not windows.",
        )
        self._keyframe_select.setEnabled(method == "keyframe")
        self._negative_pool.setEnabled(method in _PERFRAME_METHODS)

    def _set_type_field(self, name: str, enabled: bool, reason: str) -> None:
        _label, widget = self._fields[name]
        widget.setEnabled(enabled)
        widget.setToolTip(
            field_meta(QueryEvalConfig, name)["help"] if enabled else reason
        )
        label = self._form.label_for(widget)
        if label is not None:
            label.setEnabled(enabled)

    # -- config ------------------------------------------------------------- #
    def config(self) -> dict:
        cfg = {
            "embed_run_id": self._embed.currentData(),
            "behaviors": self._checked_behaviors(),
            "stream_name": self._stream.currentData(),
            "fusion": self._fusion.currentText(),
            "method": self._method.currentText(),
            "keyframe_select": self._keyframe_select.currentText(),
            "negative_pool": self._negative_pool.currentText(),
            # 0 → the subject's own frame rate decides the split
            "trajectory_split_gap_s": optional_value(self._split_gap),
            "max_results": optional_value(self._max_results),
            # unset → the holdout *is* the query, which is the original behaviour
            "n_holdout": optional_value(self._n_holdout),
        }
        for name, (_, widget) in self._fields.items():
            cfg[name] = widget.value()
        return cfg

    def validation_error(self) -> str | None:
        if self._embed.currentData() is None:
            return (
                "No completed embedding run is common to the selected recordings — run "
                "embedding first, or pick recordings that share one."
            )
        if not self._checked_behaviors():
            return "Check at least one behavior to score (the recordings' annotations declare which)."
        if (
            self._fields["folds"][1].value() == 0
            and self._fields["n_exemplars"][1].value() != 1
        ):
            return "Leave-one-out (folds=0) uses each instance as one query — set exemplars to 1."
        return None

    def load_config(self, config: dict) -> None:
        """Pre-fill from a previous run's config — for Refine and the settings picker.

        The argument is the flat config dict ``config()`` returns (the dialog merges the
        record's ``params`` over its promoted top-level keys), so the two are inverses and
        ``tests/gui/test_config_round_trip.py`` can say so.
        """
        idx = self._embed.findData(config.get("embed_run_id"))
        if idx >= 0:
            self._embed.setCurrentIndex(idx)
        by_name = {name: widget for name, (_, widget) in self._fields.items()}
        by_name["track_id"].setValue(subject_key(config.get("track_id")))
        if config.get("fusion"):
            self._fusion.setCurrentText(config["fusion"])
        if config.get("method"):
            self._method.setCurrentText(config["method"])
        if config.get("keyframe_select"):
            self._keyframe_select.setCurrentText(config["keyframe_select"])
        # Not a cosmetic knob: `labeled_clean` is the discredited negative pool kept only so
        # runs recorded before the fix stay reproducible (CLAUDE.md, "labels score a method").
        # A Refine that silently reset it to the default would make an old run's number
        # un-reproducible, which is the one thing keeping the option around is for.
        if config.get("negative_pool"):
            self._negative_pool.setCurrentText(config["negative_pool"])
        for key in _SCALARS:
            if config.get(key) is not None:
                by_name[key].setValue(config[key])
        # A run recorded before the rename carries the old key.
        # A run recorded before the rename carries the old key. `0` is a real value here — a
        # split gap of zero never splits — so this turns on the key being *present*, not on
        # the value being truthy, and goes through `set_optional_value` because `None` is the
        # spin's "auto" position rather than a number.
        if "trajectory_split_gap_s" in config or "max_gap_s" in config:
            split_gap = config.get("trajectory_split_gap_s", config.get("max_gap_s"))
            set_optional_value(
                self._split_gap, None if split_gap is None else float(split_gap)
            )
        if config.get("max_results"):
            self._max_results.setValue(int(config["max_results"]))
        # Check exactly the behaviors the run scored (its recordings are already re-checked,
        # so on_recordings_changed populated the vocabulary before this runs).
        scored = {
            b["behavior"] for b in config.get("behaviors", []) if not b.get("skipped")
        }
        for i in range(self._behaviors.count()):
            item = self._behaviors.item(i)
            item.setCheckState(
                Qt.CheckState.Checked
                if item.text() in scored
                else Qt.CheckState.Unchecked
            )
        self._update_method_fields()  # re-gate for the loaded matcher


@register_analysis_module_ui
class QueryEvalModuleUI(ModuleUI):
    module_id = "query_eval"
    purpose = "How well the query finds labelled behavior: metrics and PR curves."

    def config_widget(self) -> ConfigWidget:
        return _QueryEvalConfig()

    def result_widget(self, experiment, run_record: dict) -> QWidget:
        """The inline report (metrics mean ± std + PR curves + per-fold breakdown); its
        **Explore clips…** button opens the resizable timeline / scrub / clip-player window.
        """
        return _QueryEvalResult(experiment, run_record)


# --------------------------------------------------------------------------- #
# Result view: the inline report + the resizable clip explorer window.
# --------------------------------------------------------------------------- #
# One chart-theme hue per timeline lane — the ground-truth vs prediction outcomes. By *name*
# (`ChartTheme.hue`), because each colour carries a meaning (a hit is green, a false alarm
# vermillion) that must survive the palette's slots being reordered.
_LANES = ("query", "GT", "TP", "FP", "FN")
_LANE_HUES = {
    "query": "purple",  # the held-out exemplar
    "GT": "sky",  # scored ground truth
    "TP": "green",
    "FP": "vermillion",
    "FN": "orange",
}


# The frame/scrub seam raises these when a recording's video or an upstream run is missing
# or unreadable — e.g. the tracking run an embed run points at was deleted after the eval
# was recorded (a KeyError in ClipQueryPreview._scrub_ctx). The explorer stays decode-safe:
# an unreadable recording shows empty panes, it doesn't crash the dialog.
_FRAME_ERRORS = (FileNotFoundError, ValueError, IndexError, KeyError, OSError)


def _span_text(start_t_ns: int, end_t_ns: int) -> str:
    from detail.gui.timefmt import format_time

    if end_t_ns <= start_t_ns:
        return format_time(start_t_ns)  # a point
    return f"{format_time(start_t_ns)}–{format_time(end_t_ns)}"


def _metrics_table(record: dict, by_behavior: dict) -> QTableView:
    table, model = _read_only_table(
        ["Behavior", "Type", "Folds", "AP", "±", "P", "R", "best-F1"]
    )
    for b in record.get("behaviors", []):
        m = by_behavior.get(b["behavior"])
        if m is None:  # skipped — show why
            model.appendRow(
                [
                    QStandardItem(b["behavior"]),
                    QStandardItem(b.get("behavior_type", "")),
                    QStandardItem("—"),
                    QStandardItem(f"skipped: {b.get('skipped', '')}"),
                ]
                + [QStandardItem("") for _ in range(4)]
            )
        else:
            model.appendRow(
                [
                    QStandardItem(m.behavior),
                    QStandardItem(m.behavior_type),
                    QStandardItem(str(m.n_folds)),
                    num_cell(m.ap, f"{m.ap:.3f}"),
                    num_cell(m.ap_std, f"±{m.ap_std:.3f}"),
                    num_cell(m.precision, f"{m.precision:.3f}"),
                    num_cell(m.recall, f"{m.recall:.3f}"),
                    num_cell(m.best_f1, f"{m.best_f1:.3f}"),
                ]
            )
    return table


class _SpanStrip(QWidget):
    """Interval-span timeline: one lane per outcome (query / GT / TP / FP / FN), each
    span painted as a bar in ``t_ns`` space, with a **draggable** white playhead. Click
    or drag emits the ``t_ns`` under the cursor (``seeked``). Mirrors ``_EventStrip`` but
    draws spans (like ``_TrackTimeline``) instead of point ticks."""

    seeked = pyqtSignal(int)  # the t_ns at the cursor
    _ROW_H = 16
    _LABEL_W = 56

    def __init__(self, lanes: dict, t_min: int, t_max: int) -> None:
        super().__init__()
        self._lanes = lanes  # name -> [(start_t_ns, end_t_ns), ...]
        self._t_min = int(t_min)
        self._t_max = max(int(t_max), int(t_min) + 1)
        self._cursor_t = self._t_min
        self.setMinimumWidth(360)
        self.setFixedHeight(len(_LANES) * self._ROW_H + 4)

    def set_cursor_t(self, t_ns: int) -> None:
        self._cursor_t = int(t_ns)
        self.update()

    def _plot_w(self) -> int:
        return max(self.width() - self._LABEL_W, 1)

    def _x_for(self, t_ns: int) -> int:
        frac = (t_ns - self._t_min) / (self._t_max - self._t_min)
        return self._LABEL_W + int(frac * self._plot_w())

    def _t_at(self, x: float) -> int:
        frac = min(max((x - self._LABEL_W) / self._plot_w(), 0.0), 1.0)
        return int(self._t_min + frac * (self._t_max - self._t_min))

    def mousePressEvent(self, event) -> None:
        self.seeked.emit(self._t_at(event.position().x()))

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:  # drag = scrub
            self.seeked.emit(self._t_at(event.position().x()))

    def paintEvent(self, _event) -> None:
        theme = plot_theme.current()
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(theme.surface))
        metrics = painter.fontMetrics()
        for r, name in enumerate(_LANES):
            y = r * self._ROW_H + 2
            painter.setPen(QColor(theme.text_secondary))
            painter.drawText(
                2,
                y + self._ROW_H - 4,
                metrics.elidedText(
                    name, Qt.TextElideMode.ElideRight, self._LABEL_W - 6
                ),
            )
            color = QColor(theme.hue(_LANE_HUES[name]))
            for start, end in self._lanes.get(name, ()):
                x0 = self._x_for(start)
                x1 = self._x_for(end)
                painter.fillRect(x0, y, max(x1 - x0, 2), self._ROW_H - 3, color)
        # The playhead in ink: a position, drawn over every lane, in whichever theme.
        painter.setPen(QColor(theme.text))
        cursor_x = self._x_for(self._cursor_t)
        painter.drawLine(cursor_x, 0, cursor_x, h)


class _QueryEvalResult(QWidget):
    """The eval run's inline report: per-behavior metrics (mean ± std), a behavior selector
    driving its per-fold PR curves + breakdown + summary, and an **Explore clips…** button
    that opens the resizable clip explorer. Reads through ``QueryEvalPreview`` (Tenet 2).
    """

    def __init__(self, experiment, record: dict) -> None:
        super().__init__()
        from detail.api.run_preview import QueryEvalPreview

        self._experiment = experiment
        self._record = record
        self._run_id = record.get("id", "")
        self._eval = QueryEvalPreview(experiment.dir)
        self._explore: QDialog | None = None  # keep the non-modal window alive

        try:
            metrics = self._eval.read_metrics(self._run_id)
        except (FileNotFoundError, ValueError):
            metrics = []
        self._by_behavior = {m.behavior: m for m in metrics}

        layout = QVBoxLayout(self)
        params = record.get("params", {})
        layout.addWidget(
            QLabel(
                f"embed {record.get('embed_run_id', '?')} · track "
                f"{record.get('track_id', '?')} · matcher {params.get('method', 'dtw')} · "
                f"fusion {params.get('fusion', '?')} · "
                f"band {params.get('band_s', '?')}s · IoU≥{params.get('iou_threshold', '?')} · "
                f"n_ex {params.get('n_exemplars', '?')} · seed {params.get('seed', '?')}"
            )
        )
        layout.addWidget(_metrics_table(record, self._by_behavior))

        scored = [m.behavior for m in metrics]
        if not scored:
            layout.addWidget(QLabel("(no behavior scored)"))
            layout.addStretch()  # top-justify this degenerate state (gui_design §3)
            return

        row = QHBoxLayout()
        self._behavior_sel = QComboBox()
        self._behavior_sel.addItems(scored)
        row.addWidget(QLabel("Behavior:"))
        row.addWidget(self._behavior_sel)
        row.addStretch()
        explore = QPushButton("Explore clips")
        explore.clicked.connect(self._open_explore)
        row.addWidget(explore)
        layout.addLayout(row)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._plot = plot_theme.themed_plot()
        self._plot.setLabel("bottom", "recall")
        self._plot.setLabel("left", "precision")
        self._plot.setXRange(0, 1)
        self._plot.setYRange(0, 1)
        bound_axes(self._plot, x=(0.0, 1.0), y=(0.0, 1.0))
        layout.addWidget(self._plot, 1)
        # Each headline beside its null — a value that doesn't clear its null isn't real.
        layout.addWidget(
            QLabel(
                "By overlap strictness — (null) is the same result with the answer "
                "destroyed: predictions slid elsewhere for recall, ranking scrambled for AP."
            )
        )
        self._overlap_table, self._overlap_model = _read_only_table(
            ["Overlap", "Recall", "(null)", "lift", "P", "AP", "(null)", "lift"]
        )
        self._overlap_table.setMaximumHeight(140)
        layout.addWidget(self._overlap_table)
        self._fold_table, self._fold_model = _read_only_table(
            ["Fold", "n_gt", "TP", "FP", "FN", "AP", "P", "R", "best-F1"]
        )
        layout.addWidget(self._fold_table)

        self._behavior_sel.currentTextChanged.connect(self._show_behavior)
        self._show_behavior(scored[0])

    def _show_behavior(self, behavior: str) -> None:
        m = self._by_behavior.get(behavior)
        if m is None:
            return
        self._summary.setText(
            f"AP {m.ap:.3f} ±{m.ap_std:.3f}  ·  P {m.precision:.3f}  ·  R {m.recall:.3f}"
            f"  ·  best-F1 {m.best_f1:.3f}   |   GT {m.n_gt} · Pred {m.n_pred} · "
            f"TP {m.tp} · FP {m.fp} · FN {m.fn}   |   {m.n_folds} fold(s)"
        )
        self._plot.clear()
        try:
            per_fold = self._eval.read_pr(self._run_id, behavior)
        except (FileNotFoundError, ValueError):
            per_fold = {}
        # One thin translucent line per fold: several runs of one method, not several series.
        fold_colour = plot_theme.qcolor(plot_theme.current().identity(0), alpha=150)
        for _fold, points in sorted(per_fold.items()):
            if points:
                self._plot.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    pen=plot_theme.pen(
                        fold_colour, width=plot_theme.MARKS.thin_line_width
                    ),
                )
        self._overlap_model.removeRows(0, self._overlap_model.rowCount())
        try:
            overlap = self._eval.read_overlap(self._run_id, behavior)
        except (FileNotFoundError, ValueError):
            overlap = []  # a run recorded before nulls existed has no overlap.csv
        for o in overlap:
            self._overlap_model.appendRow(
                [num_cell(o.iou, f"{o.iou:.2f}")]
                + [
                    num_cell(v, f"{v:.3f}")
                    for v in (
                        o.recall,
                        o.recall_null,
                        o.recall_lift,
                        o.precision,
                        o.ap,
                        o.ap_null,
                        o.ap_lift,
                    )
                ]
            )
        self._fold_model.removeRows(0, self._fold_model.rowCount())
        try:
            folds = self._eval.read_folds(self._run_id, behavior)
        except (FileNotFoundError, ValueError):
            folds = []
        for f in folds:
            self._fold_model.appendRow(
                [QStandardItem(str(x)) for x in (f.fold, f.n_gt, f.tp, f.fp, f.fn)]
                + [
                    num_cell(v, f"{v:.3f}")
                    for v in (f.ap, f.precision, f.recall, f.best_f1)
                ]
            )

    def _open_explore(self) -> None:
        if (
            self._explore is not None
        ):  # already open → one window; bring it to the front
            self._explore.raise_()
            self._explore.activateWindow()
            return
        dialog = _QueryEvalExploreDialog(
            self._experiment,
            self._record,
            self._by_behavior,
            self._behavior_sel.currentText(),
            parent=self,
        )
        # Closing destroys it (stopping the clip player's autoplay timer) and clears the
        # ref so a later click opens a fresh one rather than raising a dead window.
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.destroyed.connect(lambda: setattr(self, "_explore", None))
        self._explore = dialog
        dialog.show()  # non-modal + resizable, so the report stays reachable


class _QueryEvalExploreDialog(Dialog):
    """A resizable window to review the eval's clips. Behavior / recording / fold selectors
    over a GT/TP/FP/FN **timeline**, the **whole-recording scrubber** + frame/mask, a reused
    clip **player** (source over the pose mask, Play), and a filterable clips table —
    all splitter-resizable. Selecting a clip (a table row or a timeline span) scrubs the
    recording there **and** plays the clip. Reads via ``QueryEvalPreview`` + ``ClipQueryPreview``
    (Tenet 2); decode-safe (an unreadable recording shows empty panes)."""

    def __init__(
        self, experiment, record: dict, by_behavior: dict, behavior: str, parent=None
    ):
        super().__init__(parent)
        from detail.api.run_preview import ClipQueryPreview, QueryEvalPreview
        from detail.gui.behavior_query import _ClipPlayer
        from detail.gui.image_view import ImageView, PictureWithControls
        from detail.gui.scrubber import FrameScrubber

        self.setWindowTitle(f"Explore clips — {record.get('id', '')}")
        self._record = record
        self._run_id = record.get("id", "")
        self._embed = record.get("embed_run_id", "")
        self._track = coerce_series_key(record.get("track_id", 0))
        self._eval = QueryEvalPreview(experiment.dir)
        self._frames = ClipQueryPreview(
            experiment.dir
        )  # frame/scrub seam, same embed run
        self._by_behavior = by_behavior
        self._preds: dict[str, list] = {}  # behavior -> PredictionRow[]  (lazy)
        self._gt: dict[str, list] = {}  # behavior -> GroundTruthRow[]   (lazy)
        self._cur = (
            "",
            "",
            0,
        )  # (behavior, recording, fold) for the off-thread decoder
        self._rows: list = []  # clips-table rows, aligned with the table

        root = self.body()
        sel = QHBoxLayout()
        self._behavior_sel = QComboBox()
        self._behavior_sel.addItems(sorted(by_behavior))
        self._recording_sel = QComboBox()
        self._fold_sel = QComboBox()
        self._filter = QComboBox()
        self._filter.addItems(["All", "TP", "FP", "FN", "Query"])
        for label, combo in (
            ("Behavior:", self._behavior_sel),
            ("Recording:", self._recording_sel),
            ("Fold:", self._fold_sel),
            ("Clips:", self._filter),
        ):
            sel.addWidget(QLabel(label))
            sel.addWidget(combo)
        sel.addStretch()
        root.addLayout(sel)

        # timeline (overview) — rebuilt per (behavior, recording, fold)
        self._strip_holder = QVBoxLayout()
        self._strip: _SpanStrip | None = None
        strip_w = QWidget()
        strip_w.setLayout(self._strip_holder)

        # whole-recording scrub + frame | mask
        # A read-only frame pane beside the TP/FP/FN timeline — a movie.
        self._scrubber = FrameScrubber(playable=True)
        self._scrubber.frame_changed.connect(self._on_scrub)
        self._scrubber.set_decoder(self._decode)
        self._scrubber.rendered.connect(self._display)
        self._src = ImageView(zoomable=True)
        from detail.gui.busy import decode_overlay

        self._decode_busy = decode_overlay(self._scrubber, self._src)
        # One pane. The second showed the embedder's input image — a question about the method
        # beside footage you are watching to judge a retrieval — and it halved the height the
        # video had on a page that is already four panes deep. `Masks` puts the same subject
        # marking on the frame itself.
        self._masks = QCheckBox("Masks")
        self._masks.setToolTip(
            "Tint the subject's mask onto the frame. About twice the cost a frame, so playback "
            "is slower with it on."
        )
        self._masks.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # On by default, the tool's default for every player — set before `toggled` is wired.
        self._masks.setChecked(True)
        self._masks.toggled.connect(
            lambda _on: self._scrubber.set_frame(self._scrubber.frame())
        )
        self._scrubber.add_control(self._masks)
        # Picture directly above its controls, centred in whatever room this panel is given
        # (#396) — not the picture claiming the whole panel and the controls sinking to
        # wherever it happens to end.
        # `top=True`, like the players beside a list on Browse and the audit tab: these two
        # share a splitter row with the clips table, and a splitter hands every child the whole
        # height. Centred in that, both pictures sat halfway down a pane whose neighbour starts
        # at the top, and the clips table drew as many rows as the dialog was tall.
        self._scrub_block = PictureWithControls(self._src, self._scrubber, top=True)
        scrub_panel = QWidget()
        scrub_box = QVBoxLayout(scrub_panel)
        scrub_box.setContentsMargins(0, 0, 0, 0)
        scrub_box.addWidget(QLabel("Recording scrub"))
        scrub_box.addWidget(self._scrub_block, 1)

        # the reused clip player (plays the selected clip's span)
        self._player = _ClipPlayer(
            experiment, masked=True, autoplay=True, context_s=0, top=True
        )
        player_panel = QWidget()
        player_box = QVBoxLayout(player_panel)
        player_box.setContentsMargins(0, 0, 0, 0)
        player_box.addWidget(QLabel("Play clip"))
        player_box.addWidget(self._player, 1)

        # clips table (filter drives which outcomes are listed)
        self._clips_table, self._clips_model = _read_only_table(
            ["Rank", "Outcome", "Span", "Score", "IoU"]
        )
        self._clips_table.selectionModel().currentRowChanged.connect(self._on_clip_row)
        # Now, while it is empty, and again on every refill — a table first sized after it has
        # been laid out never takes the size (`fit_table_height`). `min_rows=0`: it sits beside
        # the players in a splitter, which decides the share.
        fit_table_height(self._clips_table, max_rows=BROWSE_ROWS, min_rows=0)
        clips_panel = QWidget()
        clips_box = QVBoxLayout(clips_panel)
        clips_box.setContentsMargins(0, 0, 0, 0)
        clips_box.addWidget(QLabel("Clips"))
        clips_box.addWidget(top_of_pane(self._clips_table), 1)

        # [clips | player | recording-scrub], side by side; the clips fold by their grip to a
        # strip reading "Clips", the two players do not.
        hsplit = GripSplitter(Qt.Orientation.Horizontal, key="query_eval_explore")
        hsplit.addWidget(clips_panel)
        hsplit.addWidget(player_panel)
        hsplit.addWidget(scrub_panel)
        hsplit.set_foldable(0, "Clips")
        if not hsplit.remember():
            hsplit.setSizes([320, 380, 400])
        # The timeline above them is a section, not a pane (`tests/gui/test_workspace_parity.py`:
        # vertical sections are bands). It folds by its header; it has no drag bar because the
        # strip is a fixed height, and a bar that resizes nothing reads as broken.
        root.addWidget(CollapsibleBand("Timeline", strip_w, key="query_eval_timeline"))
        root.addWidget(hsplit, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)

        self._behavior_sel.currentTextChanged.connect(self._on_behavior)
        self._recording_sel.currentTextChanged.connect(lambda _t: self._rebuild())
        self._fold_sel.currentIndexChanged.connect(lambda _i: self._rebuild())
        self._filter.currentTextChanged.connect(lambda _t: self._rebuild_clips())
        idx = self._behavior_sel.findText(behavior)
        if idx >= 0:
            self._behavior_sel.setCurrentIndex(idx)
        self._on_behavior(self._behavior_sel.currentText())

    # -- data (lazy, per behavior) ------------------------------------------ #
    def _behavior_preds(self, behavior: str) -> list:
        if behavior not in self._preds:
            try:
                self._preds[behavior] = self._eval.read_predictions(
                    self._run_id, behavior
                )
            except (FileNotFoundError, ValueError):
                self._preds[behavior] = []
        return self._preds[behavior]

    def _behavior_gt(self, behavior: str) -> list:
        if behavior not in self._gt:
            try:
                self._gt[behavior] = self._eval.read_ground_truth(
                    self._run_id, behavior
                )
            except (FileNotFoundError, ValueError):
                self._gt[behavior] = []
        return self._gt[behavior]

    # -- behavior → repopulate recording/fold, then rebuild ------------------- #
    def _on_behavior(self, behavior: str) -> None:
        m = self._by_behavior.get(behavior)
        if m is None:
            return
        rows = self._behavior_preds(behavior) + self._behavior_gt(behavior)
        recordings = sorted({r.recording_id for r in rows})
        n_folds = m.n_folds or 1
        self._recording_sel.blockSignals(True)
        self._fold_sel.blockSignals(True)
        self._recording_sel.clear()
        self._recording_sel.addItems(recordings)
        self._fold_sel.clear()
        self._fold_sel.addItems([str(i) for i in range(n_folds)])
        self._recording_sel.blockSignals(False)
        self._fold_sel.blockSignals(False)
        self._rebuild()

    # -- (behavior, recording, fold) → timeline + scrubber + clips ------------ #
    def _rebuild(self) -> None:
        behavior = self._behavior_sel.currentText()
        recording = self._recording_sel.currentText()
        fold = self._fold_sel.currentIndex()
        if not behavior or not recording:
            return
        self._cur = (behavior, recording, fold)
        preds = [
            p
            for p in self._behavior_preds(behavior)
            if p.recording_id == recording and p.fold == fold
        ]
        gt = [
            g
            for g in self._behavior_gt(behavior)
            if g.recording_id == recording and g.fold == fold
        ]
        lanes = {
            "query": [(g.start_t_ns, g.end_t_ns) for g in gt if g.role == "exemplar"],
            "GT": [(g.start_t_ns, g.end_t_ns) for g in gt if g.role == "scored"],
            "TP": [(p.start_t_ns, p.end_t_ns) for p in preds if p.outcome == "tp"],
            "FP": [(p.start_t_ns, p.end_t_ns) for p in preds if p.outcome == "fp"],
            "FN": [
                (g.start_t_ns, g.end_t_ns)
                for g in gt
                if g.role == "scored" and g.outcome == "fn"
            ],
        }
        try:
            n_frames = self._frames.recording_frame_count(recording, self._embed)
            t_max = (
                self._frames.frame_t_ns(recording, self._embed, n_frames - 1)
                if n_frames
                else max((e for spans in lanes.values() for _s, e in spans), default=1)
            )
        except _FRAME_ERRORS:
            n_frames, t_max = 0, max(
                (e for spans in lanes.values() for _s, e in spans), default=1
            )

        if self._strip is not None:
            self._strip_holder.removeWidget(self._strip)
            self._strip.deleteLater()
        self._strip = _SpanStrip(lanes, 0, t_max)
        self._strip.seeked.connect(self._on_seek)
        self._strip_holder.addWidget(self._strip)

        self._scrubber.set_total(n_frames)
        try:
            t_ns = self._frames.timestamps(recording, self._embed)
        except _FRAME_ERRORS:
            t_ns = None  # no time shown, rather than one guessed from a frame rate
        if t_ns is not None and len(t_ns) == n_frames:
            self._scrubber.set_timestamps(t_ns)
        try:
            self._scrubber.set_playback_rate(
                self._frames.frame_rate(recording, self._embed)
            )
        except ValueError:
            # An eval run outlives the embedding it scored: `frame_rate` needs the scrub
            # context and refuses. The frames themselves fail the same way, but through the
            # decode path, which reports it off-thread. A playback rate is a display value with
            # a working default, so it must not be the thing that stops the dialog opening.
            pass
        self._rebuild_clips(preds, gt)
        self._scrubber.set_frame(0)

    def _rebuild_clips(self, preds=None, gt=None) -> None:
        if preds is None or gt is None:
            behavior, recording, fold = self._cur
            preds = [
                p
                for p in self._behavior_preds(behavior)
                if p.recording_id == recording and p.fold == fold
            ]
            gt = [
                g
                for g in self._behavior_gt(behavior)
                if g.recording_id == recording and g.fold == fold
            ]
        want = self._filter.currentText()
        # rows: (rank, outcome, start_t_ns, end_t_ns, score, iou) — predictions (TP/FP) +
        # missed GT (FN) + the held-out query exemplar(s), all playable.
        rows = []
        for p in preds:
            if want in ("All", p.outcome.upper()):
                rows.append((p.rank, p.outcome.upper(), p.start_t_ns, p.end_t_ns,
                             p.score, p.iou))  # fmt: skip
        if want in ("All", "FN"):
            for g in gt:
                if g.role == "scored" and g.outcome == "fn":
                    rows.append((None, "FN", g.start_t_ns, g.end_t_ns, None, None))
        if want in ("All", "Query"):
            for g in gt:
                if g.role == "exemplar":
                    rows.append((None, "Query", g.start_t_ns, g.end_t_ns, None, None))
        rows.sort(key=lambda r: r[2])  # by time
        self._rows = rows
        self._clips_model.removeRows(0, self._clips_model.rowCount())
        for index, (rank, outcome, start, end, score, iou) in enumerate(rows):
            rank_item = QStandardItem("" if rank is None else str(rank))
            # Identity is the row's own index into `self._rows` — stamped so
            # `_on_clip_row` survives a sort instead of trusting the clicked row's
            # position.
            stamp_row_id(rank_item, index)
            self._clips_model.appendRow(
                [
                    rank_item,
                    QStandardItem(outcome),
                    QStandardItem(_span_text(start, end)),
                    num_cell(score, "" if score is None else f"{score:.3f}"),
                    num_cell(iou, "" if iou is None else f"{iou:.2f}"),
                ]
            )
        fit_table_height(self._clips_table, max_rows=BROWSE_ROWS, min_rows=0)

    # -- timeline ↔ scrubber ↔ frame; row → scrub + play -------------------- #
    def _on_seek(self, t_ns: int) -> None:
        _behavior, recording, _fold = self._cur
        try:
            frame = self._frames.t_ns_to_frame(recording, self._embed, t_ns)
        except _FRAME_ERRORS:
            return
        self._scrubber.set_frame(frame)

    def _on_scrub(self, frame: int) -> None:
        _behavior, recording, _fold = self._cur
        if self._strip is None:
            return
        try:
            self._strip.set_cursor_t(
                self._frames.frame_t_ns(recording, self._embed, frame)
            )
        except _FRAME_ERRORS:
            pass

    def _decode(self, frame: int):
        # Off the UI thread (no Qt): decode the scrub frame for the current recording. The
        # (recording, embed, track) context is captured on the UI thread in _cur.
        _behavior, recording, _fold = self._cur
        if not recording:
            return None
        try:
            reader = (
                self._frames.frame_masked
                if self._masks.isChecked()
                else self._frames.frame_plain
            )
            return frame_source.decode(  # off-thread (scrubber decoder)
                lambda: reader(recording, self._embed, self._track, frame)
            )
        except _FRAME_ERRORS:
            return None

    def _display(self, _frame: int, payload) -> None:
        from detail.gui.run_preview import _to_pixmap

        if payload is None:
            self._src.clear_frame("(no frame)")
            return
        view, reason = payload
        if reason is not None:
            self._src.clear_frame(frame_source.unavailable_text(reason))
            return
        rgb, _actual, _t_ns = view
        self._src.set_frame(_to_pixmap(rgb))
        self._scrub_block.set_aspect(self._src.pixmap_size())

    def _on_clip_row(self, current, _previous) -> None:
        key = row_id(current)
        if key is None or not (0 <= key < len(self._rows)):
            return
        rank, outcome, start, end, score, iou = self._rows[key]
        self._on_seek(start)  # scrub the recording (+ move the timeline cursor)
        self._play_clip(start, end, outcome, score, rank)

    def _play_clip(self, start, end, outcome, score, rank) -> None:
        from detail.gui.behavior_query import ClipSpec

        _behavior, recording, _fold = self._cur
        if not recording:
            return
        self._player.set_clip(
            ClipSpec(
                recording,
                self._track,
                self._embed,
                start,
                end,
                f"{outcome} · {_span_text(start, end)}",
                score,
                rank,
            )
        )
