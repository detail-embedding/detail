"""Query ("Behavior scan") analysis module UI (§20).

The launch form: pick the embedding run common to the searched recordings, then build a
**Behaviors** table — each behavior is a name plus the example clips that define it, marked
in ``ClipExampleDialog``, which takes every behavior's examples and non-examples in one pass. The method sits in advanced settings, because
without ground truth a user cannot evaluate one matcher against another on their own data.

Three things this form does *not* have, each on purpose:

- **No threshold.** The search stores no cutoff; choosing one is calibration, which happens
  after the search and costs nothing to redo. A cutoff on this form would be the thing §20
  removed.
- **No per-behavior parameters.** Instances found under different settings are not comparable
  and one run emits one artifact, so the method's params are run-level. Two settings means two
  runs.
- **No per-behavior parameters on the *form*.** Instances found under different settings are
  not comparable and one run emits one artifact, so the method's params are run-level.

``result_widget`` returns the **calibration workspace** — a behavior selector over Calibrate and
Browse. It is a stateful result view that writes, which ``docs/gui_design.md`` §3.2 allows for an
authored run (§21) and which the removed ``clip_annotation`` established. Audit is the third tab §20
specifies and is not built yet; §21 makes it generic over the artifact rather than Query's, so it
arrives as a shared widget rather than as a tab grown here.
"""

import dataclasses

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabBar,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QPalette,
    QStandardItem,
    QStandardItemModel,
)

from detail.configs.query import DEFAULT_METHOD, QueryConfig
from detail.core import chart_theme
from detail.gui.forms import ConfigForm
from detail.gui.grip_splitter import GripSplitter
from detail.gui.help_ui import field_label, help_button, label_with
from detail.gui.modules._common import common_complete_run_items, repopulate_combo
from detail.gui._pydantic_form import (
    field_widget,
    optional_spin,
    set_optional_value,
    optional_value,
)
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    MultiViewWorkspace,
    register_analysis_module_ui,
)
from detail.gui.busy import run_busy, run_progress
from detail.gui.clip_grid import ConfidenceStrip, score_digits
from detail.gui.confidence import ConfidenceRow, showing_caption
from detail.gui.audit_tab import AuditTab
from detail.gui.cluster_strip import EthogramStrip, ruled
from detail.gui.collapse import CollapsibleBand
from detail.gui.measurements import BASELINE_RECORDING, MeasurementsPanel
from detail.gui.tables import (
    BROWSE_ROWS,
    num_cell,
    SortCell,
    enable_sorting,
    fit_table_height,
    make_columns_resizable,
    row_id,
    top_of_pane,
    stamp_row_id,
)
from detail.gui.sizing import lines, remember_size, remembered_size
from detail.gui.spatial_map import SpatialMap, frame_aspect
from detail.gui.method_form import MethodFields
from detail.gui.example_spread import spread_text
from detail.gui import plot_theme
from detail.gui.plots import bound_axes
from detail.gui.sizing import lines
from detail.gui.tables import fill_width
from detail.pipeline.query import QueryModule, resolve_method
from detail.gui.status_color import WARN
from detail.gui.sizing import mono_font

# §20's exemplar guardrails, as warnings rather than refusals: a user who knows their behavior
# is brief must still be able to mark a brief clip. Very short clips and a single exemplar
# both make results worse and less steady — so both are worth saying and neither is worth
# blocking on. What a *method* requires is a refusal, and lives in `QueryModule.validate`.
MIN_CLIP_S = 2.0
ADVISED_CLIPS = 3


def _query_methods() -> dict[str, type]:
    """The §20 methods this build offers, imported. Passed to ``MethodFields`` whole, so a
    method registered tomorrow gets its parameters on this form without touching it.

    §25's filter belongs here — the one place the form *enumerates* what a user may pick —
    and not at ``resolve_method``, which a completed run's result view calls to render itself.
    Hiding governs launching, not reading.
    """
    from detail import surfaces
    from detail.core.registries import QUERY_METHODS
    from detail.query_methods import import_query_methods

    import_query_methods()
    return {
        method_id: cls
        for method_id, cls in QUERY_METHODS.items()
        if surfaces.is_surfaced(method_id)
    }


def _method_ids() -> list[str]:
    return sorted(_query_methods())


class _QueryConfig(ConfigWidget):
    def __init__(self) -> None:
        super().__init__()
        self._behaviors: list[dict] = (
            []
        )  # {"name", "examples": [...], "negatives": [...]}
        #: Set when `load_config` was handed a method this build does not offer. Held rather
        #: than shown once, because `_refresh_advice` rebuilds the notice from scratch and a
        #: silent substitution is exactly the failure a seeded form must not have.
        self._method_note: str | None = None
        self._experiment = None
        self._recordings: list = []
        form = ConfigForm(self, QueryConfig)

        self._embed = QComboBox()
        self._embed.currentIndexChanged.connect(self.coverage_changed.emit)
        form.addRow(label_with("Embed run:", help_button("embedding-run")), self._embed)
        self._track = field_widget(QueryConfig, "track_id")
        form.addRow(field_label("query", "track_id", "Subject:"), self._track)

        self._model = QStandardItemModel()
        self._model.setHorizontalHeaderLabels(
            ["Behavior", "Examples", "Non-examples", "Spread", "Time"]
        )
        self._table = QTableView()
        fill_width(self._table)
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        make_columns_resizable(self._table)
        fill_width(self._table)  # the last column ("Rows") takes the remaining space
        enable_sorting(self._table)
        form.addRow(
            label_with("Behaviors:", help_button("behavior-query")), self._table
        )

        # Amber, word-wrapped, hidden until it applies — the pattern `modules/tracking.py`
        # uses for its confidence warning. A palette colour, not a stylesheet (GUI style).
        self._advice = QLabel()
        self._advice.setWordWrap(True)
        palette = self._advice.palette()
        palette.setColor(QPalette.ColorRole.WindowText, WARN)
        self._advice.setPalette(palette)
        self._advice.setVisible(False)
        form.addRow("", self._advice)

        # One dialog marks every behavior's examples and non-examples in a single pass over
        # the video (#628); it opens on the selected behavior.
        self._mark_examples = QPushButton("Mark examples…")
        self._mark_examples.clicked.connect(self._open_mark_examples)
        row = QHBoxLayout()
        # The only control here: behaviors are added, renamed and removed in the dialog, where
        # their clips are, so this table is the summary and the way in (#628).
        row.addWidget(self._mark_examples)
        row.addStretch()
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("", holder)

        # Main settings, not advanced: unlike the method's own knobs, a user can say what it
        # means for their behavior ("nothing under half a second") without a measurement.
        self._min_match = field_widget(QueryConfig, "min_match_s")
        self._min_match.setSpecialValueText("off")
        form.addRow(
            field_label("query", "min_match_s", "Min match length:"), self._min_match
        )

        self._method = QComboBox()
        # The form opens on `configs/query.DEFAULT_METHOD` rather than blank. §20 used to have
        # it open empty and refuse until a method was picked, on the grounds that nothing
        # separated the two; that stopped being true once one of them was measured at roughly
        # thirty times the speed of the other. An empty picker does not leave the choice open,
        # it just makes everyone make it — and the field is in advanced settings, where most
        # people will never see it.
        for method_id in _method_ids():
            self._method.addItem(method_id, method_id)
        if self._method.findData(DEFAULT_METHOD) >= 0:
            self._method.setCurrentIndex(self._method.findData(DEFAULT_METHOD))
        self._method.currentIndexChanged.connect(self._on_method_changed)
        form.addRow(field_label("query", "method", "Method:"), self._method)

        # Every registered method's parameters, the selected one's shown. Added right after
        # the picker so a knob sits under the control that decides whether it applies.
        self._method_fields = MethodFields(form, _query_methods())

        self._split_gap = optional_spin(
            QueryConfig, "trajectory_split_gap_s", "auto (~4 frames)"
        )
        form.addRow(
            field_label("query", "trajectory_split_gap_s", "Trajectory split gap:"),
            self._split_gap,
        )
        self._form = form
        self._on_method_changed()
        form.finish()
        self._refresh_table()

    # -- launch-dialog hooks ------------------------------------------------ #
    def on_experiment_set(self, experiment) -> None:
        self._experiment = experiment

    def on_recordings_changed(self, recordings: list) -> None:
        self._recordings = recordings
        repopulate_combo(
            self._embed,
            common_complete_run_items(
                recordings,
                "embedding",
                self._experiment.model_names() if self._experiment else None,
            ),
        )

    # -- behaviors ---------------------------------------------------------- #
    def _selected(self) -> int:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return -1
        key = row_id(rows[0])
        return -1 if key is None else key

    def _open_mark_examples(self) -> None:
        title = "Mark examples"
        embed = self._embed.currentData()
        if not self._recordings or not embed:
            QMessageBox.information(
                self, title, "Check the recordings to search first."
            )
            return
        from detail.gui.behavior_query import ClipExampleDialog

        # Both continuity inputs come from what this form has selected, so the picker accepts
        # exactly what the search will: the split gap from the form, and whether a clip must
        # sit inside one stretch from the *method* (`ExemplarBounds.requires_contiguous`).
        method_id = self._method.currentData()
        dialog = ClipExampleDialog(
            self._experiment,
            self._recordings,
            embed,
            parent=self,
            max_gap_ns=self._split_gap_ns(),
            require_contiguous=(
                resolve_method(method_id).exemplars.requires_contiguous
                if method_id
                else True
            ),
            behaviors=self._behaviors,
            active=max(self._selected(), 0),
            negatives_reason=self._negatives_reason(),
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._behaviors = dialog.behaviors
            self._refresh_table()

    def _on_method_changed(self, *_args) -> None:
        """Show the selected method's parameter rows, and re-judge whether non-examples are used."""
        self._method_fields.show(self._method.currentData())
        self._form.finish()  # the advanced count follows what is now on offer
        self._refresh_table()  # the non-examples column says whether they will be used
        # The user has now chosen for themselves, so the note about a seeded run's
        # unavailable method has served its purpose.
        self._method_note = None
        self._refresh_advice()

    def _negatives_reason(self) -> str | None:
        """Why the selected method cannot use non-examples, or None when it can.

        §20: a user who marks fifteen careful non-examples into a method that silently
        discards them has been misled by an empty affordance. The example dialog greys its
        non-example toggle with this reason — greyed, never hidden — and the table and the
        dialog both say that non-examples already marked will not be used.

        The reason comes from the **method**, which is the only thing that knows it. It used
        to be a sentence about scoring distance to the examples, which was true of the removed
        `dtw` and of nothing else in the registry.
        """
        method_id = self._method.currentData()
        if not method_id:
            return "Pick a method first."
        negatives = resolve_method(method_id).negatives
        if negatives.support != "unusable":
            return None
        return negatives.reason or f"{method_id} cannot use non-examples."

    def _refresh_table(self) -> None:
        self._model.removeRows(0, self._model.rowCount())
        unused_reason = self._negatives_reason()
        for index, behavior in enumerate(self._behaviors):
            rows = sum(self._span_seconds(clip) for clip in behavior["examples"])
            name_item = QStandardItem(behavior["name"])
            # Identity is the behavior's own index into `self._behaviors` — stamped so
            # `_selected` survives a sort instead of trusting the clicked row's position.
            stamp_row_id(name_item, index)
            n_neg = len(behavior["negatives"])
            negatives = num_cell(
                n_neg, f"{n_neg} (not used)" if n_neg and unused_reason else None
            )
            if n_neg and unused_reason:
                negatives.setToolTip(unused_reason)
            self._model.appendRow(
                [
                    name_item,
                    num_cell(len(behavior["examples"])),
                    negatives,
                    QStandardItem(spread_text(behavior["examples"])),
                    num_cell(rows, f"{rows:.1f} s"),
                ]
            )
        self._refresh_advice()

    @staticmethod
    def _span_seconds(clip: dict) -> float:
        if "start_t_ns" in clip and "end_t_ns" in clip:
            return (int(clip["end_t_ns"]) - int(clip["start_t_ns"])) / 1e9
        return 0.0

    def _refresh_advice(self) -> None:
        notes: list[str] = []
        if self._method_note:
            notes.append(self._method_note)
        for behavior in self._behaviors:
            clips = behavior["examples"]
            if not clips:
                continue
            if len(clips) < ADVISED_CLIPS:
                notes.append(
                    f"{behavior['name']!r} has {len(clips)} clip"
                    f"{'' if len(clips) == 1 else 's'}: 3-5 is measurably steadier"
                )
            brief = [c for c in clips if self._span_seconds(c) < MIN_CLIP_S]
            if brief:
                notes.append(
                    f"{behavior['name']!r} has {len(brief)} clip under {MIN_CLIP_S:g} s"
                )
        self._advice.setText("⚠ " + "; ".join(notes) if notes else "")
        self._advice.setVisible(bool(notes))

    # -- config ------------------------------------------------------------- #
    def _split_gap_ns(self) -> int | None:
        seconds = optional_value(self._split_gap)
        return None if seconds is None else round(seconds * 1e9)

    def load_config(self, config: dict) -> None:
        """Restore a previous scan's settings — for Refine and the settings picker.

        The **behaviors** are restored too, and they are the point: a behavior carries its
        example clips as ``(recording_id, track_id, start_t_ns, end_t_ns)``, which is the work
        a user actually wants back. That is a reference to data, but to data *inside this
        experiment* — the same experiment the picker draws its runs from — so it binds. What is
        not restored is ``embed_run_id``, which the dialog re-derives from the current
        recording selection.

        The method is selected before its parameters, since each method's rows are its own.
        """
        wanted = config.get("method")
        index = self._method.findData(wanted)
        if index >= 0:
            self._method.setCurrentIndex(index)
            self._method_note = None
        elif wanted:
            # `setCurrentIndex(-1)` and a silent skip both leave the combo on the default
            # while the row above says the form was seeded from that run — the seeded-form
            # failure `tests/gui/test_start_from.py` guards. Say which method it was and
            # what is selected instead.
            self._method_note = (
                f"that run used the {wanted!r} method, which this build does not offer; "
                f"{self._method.currentData()!r} is selected instead"
            )
        if config.get("track_id") is not None:
            self._track.setValue(int(config["track_id"]))
        self._method_fields.set_params(
            self._method.currentData(), config.get("method_params") or {}
        )
        gap_ns = config.get("trajectory_split_gap_ns")
        set_optional_value(self._split_gap, None if gap_ns is None else gap_ns / 1e9)
        if config.get("min_match_ns") is not None:
            self._min_match.setValue(int(config["min_match_ns"]) / 1e9)
        behaviors = config.get("behaviors")
        if behaviors:
            self._behaviors = [
                {
                    "name": b["name"],
                    "examples": [dict(c) for c in b.get("examples", [])],
                    "negatives": [dict(c) for c in b.get("negatives", [])],
                }
                for b in behaviors
            ]
        # Unconditional: `_refresh_table` is what usually renders the notice, and a run seeded
        # with no behaviors would otherwise swallow the note about its unavailable method.
        self._refresh_table()

    def config(self) -> dict:
        return {
            "embed_run_id": self._embed.currentData(),
            "track_id": self._track.value(),
            "method": self._method.currentData(),
            "method_params": self._method_fields.params(self._method.currentData()),
            "behaviors": [
                {
                    "name": b["name"],
                    "examples": [dict(c) for c in b["examples"]],
                    "negatives": [dict(c) for c in b["negatives"]],
                }
                for b in self._behaviors
            ],
            "trajectory_split_gap_ns": self._split_gap_ns(),
            "min_match_ns": round(self._min_match.value() * 1e9),
        }

    def validation_error(self) -> str | None:
        if self._embed.currentData() is None:
            return (
                "No completed embedding run is common to the selected recordings — run "
                "embedding first, or pick recordings that share one."
            )
        if not self._method.currentData():
            return "Pick a search method (Advanced settings)."
        if not self._behaviors:
            return "Add at least one behavior (Mark examples…)."
        for behavior in self._behaviors:
            if not behavior["examples"]:
                return (
                    f"Behavior {behavior['name']!r} has no example clips — use "
                    "Mark examples…."
                )
        return None


# How wide a band of confidence the "near cutoff" strip draws from, as a fraction of the axis.
# Points on the frequency curve. Enough that the shape is legible, few enough that building it
# stays one pass over the cached extraction.
_CURVE_STEPS = 41
# Layout A: the dial on the left at a fixed width, the evidence filling the right. Fixed
# rather than proportional so nothing on the right can resize the graph.
_DIAL_PX = 380
#: The curve's height, as a **range**: it may be squeezed to `_PLOT_MIN_PX` on a short window and
#: never grows past `_PLOT_MAX_PX`. In lines of the interface font, so a user who scales their text
#: gets a taller graph rather than the same box with bigger axis labels (`sizing.lines`).
#:
#: The floor is what a curve is still readable at. pyqtgraph's axis chrome is a *constant* ~40px
#: whatever the widget's height, so a 130px plot still spends 90px on data — measured across
#: 240/160/120/90px, the data area is 200/120/80/50. Below about eight lines the chrome is most of
#: it and the shape stops being legible.
#: Line *counts*, resolved at construction: `sizing.lines` reads the application font and
#: there is no application at import time.
_PLOT_MIN_LINES = 8
_PLOT_MAX_LINES = 17
#: Heights, for the vertical split. Distinct from `_PLAYER_PX` / `_DIAL_PX`, which are the
#: *widths* of the two panes inside the top row — reusing one number for both put 460px of
#: preferred height on the top row, pushed the sum past the window, and Qt scaled everything
#: down proportionally until the clip row was 168px and the divider's buttons were clipped.
_TOP_PX = 430
_STRIP_PX = 190
#: What a clip row cannot go below: a thumbnail, its caption, and the divider's steppers. A
#: preferred size is negotiable under pressure; this is not.
_STRIP_MIN_PX = 175
_PLAYER_PX = 460
#: The ethogram band under the clip row. Short: it answers "where did these land", which is a
#: shape, not a value anyone reads off.
_ETHOGRAM_PX = 150


def _sparkline_font() -> QFont:
    """Monospace, so the per-recording bars line up into a readable column."""
    return mono_font()


def _clip_specs(candidates, recording_id, subject, embed_run_id, timeline) -> list:
    """``QueryCandidate``s -> ``ClipSpec``s, which every clip surface in the GUI speaks.

    The candidate carries trajectory **rows**; a clip surface needs times, so the unit's
    timeline supplies them by direct indexing — the same projection the artifact writer does.
    """
    from detail.gui.behavior_query import ClipSpec

    _frame_idx, t_ns = timeline
    out = []
    for c in candidates:
        if c.start_row >= len(t_ns) or c.end_row >= len(t_ns):
            continue  # a shard from a search whose trajectory has since changed
        out.append(
            ClipSpec(
                recording_id,
                int(subject),
                embed_run_id,
                int(t_ns[c.start_row]),
                int(t_ns[c.end_row]),
                f"{recording_id} {c.confidence:.2f}",
                score=float(c.confidence),
            )
        )
    return out


class _CalibrateTab(QWidget):
    #: A cutoff was written and the artifact reprojected — the other tabs are stale.
    committed = pyqtSignal()

    """Where the cutoff is chosen: the curve, the per-recording spread, and the clips at it.

    §20's rule for this surface is that **dragging must not reproject**. The drag reads the
    cached curve and re-points the clip strip; releasing commits, off the UI thread. Conflating
    them freezes the GUI on every pixel of a drag, and the reprojection it would run per pixel
    rewrites a frame-view zarr per searched unit.
    """

    def __init__(self, run, experiment) -> None:
        super().__init__()
        import pyqtgraph as pg

        self._run = run
        self._experiment = experiment
        self._behavior_id = None
        #: Each recording's field of view, normalized — read once with the ethograms, off the
        #: UI thread. Without it a square is a bare rectangle and a dot is a coordinate rather
        #: than a place.
        self._outlines_by_recording: dict = {}
        #: Where each kept instance happened **and which instance it was**, so a clicked dot
        #: can be turned back into the clip it stands for.
        self._placements_by_recording: dict | None = {}
        #: Cached at load: `state()` is a disk read, and a clicked dot must not pay for one on
        #: the UI thread (`cluster_workspace` caches it for the same reason).
        self._embed_run_id: str = ""
        self._frame_aspects: dict = {}
        #: Built clip lists and unit timelines, per behavior / per unit. Neither changes while
        #: the run is open, and rebuilding them on every tab click was the whole of the switch
        #: lag.
        self._specs: dict[str, list] = {}
        self._timelines: dict[tuple, object] = {}
        self._curve: list = []
        self._syncing = False  # two controls drive one cutoff; do not let them echo

        from detail.gui.behavior_query import (
            _ClipPlayer,
        )  # local: avoids an import cycle

        outer = QVBoxLayout(self)
        self._header = QLabel("")
        self._header.setWordWrap(True)
        outer.addWidget(self._header)

        # Vertical: what you are looking at on top, the ordering you are cutting in the
        # middle, where it lands underneath. The clip row gets the full width because walking
        # it is the decision this screen exists for; before, it had a third of the window and
        # the dial and player had the rest.
        # Neither side folds: the clip and the count are judged together, both in view.
        top = self._top = GripSplitter(Qt.Orientation.Horizontal, key="query_calibrate")

        # -- left: the dial ------------------------------------------------------------- #
        # Its own scroll, so a cohort with many recordings cannot push the pane's contents
        # into the pane that holds the clips. The two used to share one column, and the plot
        # was its only vertically-Expanding child: every clip the strip gained or lost was
        # absorbed by the graph, which is why it moved on essentially every drag (measured at
        # 663px -> 480px across a cutoff sweep).
        dial_scroll = self._dial_scroll = QScrollArea()
        dial_scroll.setWidgetResizable(True)
        dial = QWidget()
        root = QVBoxLayout(dial)

        self._plot = plot_theme.themed_plot()
        # A **range**, not a fixed height. `setFixedHeight` is a floor and a ceiling at once, so a
        # pane shorter than the number can only crop: the widget keeps its height and Qt clips it
        # at the parent boundary, taking the x-axis and its labels with it. Measured on this pane
        # at a laptop size — a 444px dial column in a 334px viewport, cropped by 110px, of which
        # 304 was this line plus the caption's — which is the "the plots are cut off" report.
        # The pane split above is what holds the height *still* while the clips resize; this only
        # has to stop the graph inflating the column it lives in.
        self._plot.setMinimumHeight(lines(_PLOT_MIN_LINES))
        self._plot.setMaximumHeight(lines(_PLOT_MAX_LINES))
        self._plot.setLabel("bottom", "confidence cutoff")
        self._plot.setLabel("left", "instances kept")
        bound_axes(self._plot, x=(0.0, 1.0), y=(0.0, None))
        theme = plot_theme.current()
        self._line = pg.InfiniteLine(
            pos=0.5,
            angle=90,
            movable=True,
            pen=plot_theme.pen(theme.handle, width=2),
            hoverPen=plot_theme.pen(theme.handle, width=4),
        )
        # The line's position is committed to run state, and `api.query.set_cutoff` refuses a
        # value outside [0, 1] — so an unbounded drag turns a gesture into an error dialog.
        # The API check stays as the backstop; this stops the gesture reaching it.
        self._line.setBounds([0.0, 1.0])
        # Two signals, deliberately different jobs. `sigPositionChanged` fires on a drag *and*
        # on a programmatic move, so it is the view update; `sigPositionChangeFinished` fires
        # only when a real drag is released (pyqtgraph emits it inside `mouseDragEvent` on
        # `ev.isFinish()`), so it is the commit. That asymmetry is what lets the numeric box
        # below move the line without committing on every keystroke.
        self._line.sigPositionChanged.connect(self._on_line_moved)
        self._line.sigPositionChangeFinished.connect(self._on_line_released)
        self._plot.addItem(self._line)
        root.addWidget(self._plot)

        # What the curve is, in words, because the axes alone did not say. It used to be pinned
        # to a fixed height so a rewrap could not move the graph — but a word-wrapped label is
        # exactly the widget whose height depends on the width it is given, so pinning it makes
        # the *text* disappear at a narrow width instead of the label growing. 64px of it was a
        # quarter of the pane's 444px floor. It wraps; the plot's ceiling above is what stops the
        # column growing.
        legend = QLabel(
            "Each point is how many instances that cutoff returns. Left is loose, right is "
            "strict. ✕ marks a point where the method stopped storing candidates — the curve "
            "flattens there for that reason, not the animal's."
        )
        legend.setWordWrap(True)
        root.addWidget(legend)

        self._confidence = ConfidenceRow(
            heading="Match confidence cutoff",
            help_text=(
                "How confident a match must be to count as an instance. There is no ground "
                "truth to optimize against, so this is chosen by looking at what each "
                "setting returns — the curve above and the clips beside it."
            ),
        )
        self._confidence.threshold_changed.connect(self._on_confidence_typed)
        root.addWidget(self._confidence)

        root.addWidget(QLabel("Kept per recording, at this cutoff"))
        self._spread = QLabel("")
        self._spread.setFont(_sparkline_font())
        self._spread.setWordWrap(True)
        root.addWidget(self._spread)
        root.addStretch()
        dial_scroll.setWidget(dial)

        # -- right: the evidence -------------------------------------------------------- #
        # The clips are what the decision is made on, so they open visible rather than five
        # widgets down a scrolling column ("I didn't realize the matches were shown").
        self._grid = ConfidenceStrip(experiment)
        self._grid.clip_clicked.connect(self._on_clip_clicked)
        self._grid.cutoff_moved.connect(self._on_strip_cutoff_moved)
        self._grid.cutoff_released.connect(self._on_strip_cutoff_released)
        self._grid.setMinimumHeight(_STRIP_MIN_PX)

        # The scrubber spans the **clip**, and nothing either side of it. Trim mode pads the
        # window by ±3s, which is what the annotation loop needs — there you drag a match's
        # bounds outward, so you have to see past them. Here you are only deciding whether the
        # clip is the behavior, and the padding puts the playhead a third of the way along a
        # strip that is mostly not the match, which reads as starting in the middle.
        self._player = _ClipPlayer(experiment, autoplay=True, context_s=0)
        self._player.setMinimumWidth(_PLAYER_PX)

        # Top row: the clip you are judging beside the count you are judging it against. Both
        # have to be visible *while* dragging — the count is the thing the drag changes, and a
        # plot below the clip row is below the fold on a laptop.
        top.addWidget(self._player)
        top.addWidget(dial_scroll)
        dial_scroll.setMinimumWidth(_DIAL_PX)
        if not top.remember():
            top.setSizes([_PLAYER_PX, _DIAL_PX])

        # Where the kept instances actually land, one row per searched recording. Redrawn on
        # **commit**, not on drag: the artifact holds only what the cutoff kept, so there is
        # nothing to draw for a cutoff nobody has committed yet.
        ethograms = QWidget()
        ethogram_column = QVBoxLayout(ethograms)
        ethogram_column.setContentsMargins(0, 0, 0, 0)
        self._ethogram_title = QLabel("Kept instances, per recording")
        ethogram_column.addWidget(self._ethogram_title)
        self._ethograms = EthogramStrip()
        ethogram_column.addWidget(self._ethograms)
        ethogram_column.addStretch()
        # `EthogramStrip` is `setFixedHeight(rows * row_px())`, so a hundred recordings is
        # ~2,000px the page cannot squeeze — the band scrolls its rows and the drag says how
        # much you want.
        self._ethogram_scroll = QScrollArea()
        self._ethogram_scroll.setWidgetResizable(True)
        self._ethogram_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._ethogram_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._ethogram_scroll.setWidget(ethograms)
        self._ethogram_px = remembered_size("query/ethograms", lines(8))
        self._ethogram_scroll.setFixedHeight(self._ethogram_px)

        # Where those instances land, beside when they land — the same question the clustering
        # workspace's band answers, the same widget, and now the same three bands in the same
        # order. A cutoff that keeps a behavior only in one corner is a spatial bias nothing
        # else on this screen shows.
        self._map = SpatialMap(experiment)

        # **Vertical sections are bands; horizontal panes are splitters.** The player beside the
        # dials is genuinely side by side and stays a splitter; everything stacked below it folds
        # and drags, exactly as it does on the clustering page. This screen used three splitter
        # panes and no bands, which is why a comment on one page never reached the other.
        # Stretch 1, and no trailing stretch below: the top row is this page's primary content
        # (the clip you are judging beside the count you are judging it against), so it takes
        # the slack the way a fill element does. Packed at stretch 0 it was handed 302px for
        # 452px of dial and clipped the *Kept per recording* heading — the same symptom the
        # measure-and-claw-back workaround existed for, which this replaces.
        outer.addWidget(top, 1)
        self._grid_band = CollapsibleBand(
            "Clips", self._grid, key="query/clips", on_resize=self._grid.resize_by
        )
        outer.addWidget(self._grid_band)
        self._ethogram_band = CollapsibleBand(
            "Ethograms",
            ruled(self._ethograms, self._ethogram_scroll),
            key="query/ethograms",
            on_resize=self._resize_ethograms,
        )
        outer.addWidget(self._ethogram_band)
        self._map_band = CollapsibleBand(
            "Locations", self._map, key="query/locations", on_resize=self._map.resize_by
        )
        outer.addWidget(self._map_band)
        # The same band the clustering workspace carries, on the same artifact: a behavior scan's
        # matches are bouts with a speed, a duration and a path shape too, and "is this cluster
        # unusually slow" is the same question here as "is this behavior". A section on one
        # result page and not the other is what `test_workspace_parity` exists to catch.
        # Against the **recording**, not against the other behaviors: a query's matches are a
        # selection from the footage rather than a partition of it, and four of five real query
        # runs have a single behavior — so "the other behaviors" is nothing at all, and the band
        # drew nothing at all. "Faster than most of what this animal does" is the question anyway.
        self._measurements = MeasurementsPanel(
            experiment,
            "matches",
            "behavior",
            key="query/measurements",
            baseline=BASELINE_RECORDING,
        )
        # A commit rewrites the matches the band partitions, and leaves the behavior selected.
        self.committed.connect(self._measurements.recheck)
        self._measurement_band = CollapsibleBand(
            "Interpretation",
            self._measurements,
            key="query/measurements",
            on_resize=self._measurements.resize_by,
        )
        outer.addWidget(self._measurement_band)

    def showEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().showEvent(event)
        # One turn later: at `showEvent` the dial's children exist and have not been laid out,
        # so its hint is still short and measuring here reads "nothing to do".
        QTimer.singleShot(0, self._floor_the_dial)

    def _floor_the_dial(self) -> None:
        """Let the dial declare the height its own contents need.

        It replaces a method that measured the dial and clawed pane height back off a vertical
        splitter. The splitter is gone — the sections below are bands — so the honest form is a
        *minimum*: the page scrolls when the sum does not fit (`PageScroll`), where a scroll area
        with no floor simply clipped the *Kept per recording* heading and called it laid out.
        """
        inner = self._dial_scroll.widget()
        needed = inner.sizeHint().height() + 2 * self._dial_scroll.frameWidth()
        if needed > self._dial_scroll.minimumHeight():
            self._dial_scroll.setMinimumHeight(needed)

    def _resize_ethograms(self, delta: int) -> None:
        """How many recordings' timelines are on screen at once."""
        self._ethogram_px = max(lines(3), min(self._ethogram_px + delta, lines(40)))
        self._ethogram_scroll.setFixedHeight(self._ethogram_px)
        remember_size("query/ethograms", self._ethogram_px)

    def _on_strip_cutoff_moved(self, value: float) -> None:
        """The line moved in the strip: mirror it onto the graph and the box, view only."""
        if self._syncing:
            return
        self._set_cutoff_widgets(value)
        self._refresh_header(value)

    def _on_strip_cutoff_released(self, value: float) -> None:
        """The line was let go: commit, off the UI thread, exactly as a graph drag does."""
        self._commit(value)

    def _on_clip_clicked(self, clip) -> None:
        """Play the clicked clip. The cell shows a pointing-hand cursor, so a click has to do
        something; this signal used to be connected to nothing at all."""
        # The strip marks its own selected cell (it owns the mapping from clip to cell), so
        # this only has to play it.
        self._player.set_clip(clip)

    def set_behavior(self, behavior_id: str, curve: list) -> None:
        """Point the tab at a behavior whose results are already warmed."""
        self._behavior_id = behavior_id
        self._curve = list(curve)
        behavior = self._run.state().behavior(behavior_id)
        start = 0.5 if behavior.cutoff is None else float(behavior.cutoff)
        self._draw_curve()
        self._set_cutoff_widgets(start)
        self._refresh_clips()  # the whole ranked list, once per behavior
        self._refresh_view(start)
        # Measured against the recording, so "slower than most" means something. This follows the
        # behavior rather than the slider; a committed cutoff changes the matches, and reaches the
        # band through `committed`.
        self._measurements.select(self._run, behavior_id)
        # ...and on opening, not only after a commit. Hanging the ethograms off `_on_committed`
        # alone left the pane holding a heading and nothing under it until somebody happened to
        # move the cutoff — on a behavior already calibrated, that is a blank panel reporting a
        # result that exists.
        self._refresh_ethograms()

    def cutoff(self) -> float:
        return float(self._line.value())

    # -- drawing ------------------------------------------------------------------------ #

    def _draw_curve(self) -> None:
        import numpy as np
        import pyqtgraph as pg

        self._plot.clear()
        self._plot.addItem(self._line)
        if not self._curve:
            return
        x = np.array([p["cutoff"] for p in self._curve], dtype=float)
        y = np.array([p["n_instances"] for p in self._curve], dtype=float)
        # Pin both axes. The cutoff marker is a plot item, so with auto-range on, *dragging it*
        # nudged the x range — the graph moving under the hand that moved the line, which is the
        # second half of "the graph resizes as I move the confidence bar". The domain is the
        # confidence axis, [0, 1], and it is known before the data.
        self._plot.setXRange(0.0, 1.0, padding=0.02)
        self._plot.setYRange(0.0, float(max(y.max(), 1.0)), padding=0.05)
        self._plot.disableAutoRange()
        theme = plot_theme.current()
        self._plot.plot(x, y, pen=plot_theme.pen(self._curve_colour()))
        # A curve that flattens because the method stopped storing candidates looks exactly
        # like one that flattens because the animal stopped, so the points a cap bound are
        # marked rather than left to be misread (§20).
        capped_x = [p["cutoff"] for p in self._curve if p["capped"]]
        if capped_x:
            capped_y = [p["n_instances"] for p in self._curve if p["capped"]]
            self._plot.plot(
                np.array(capped_x, dtype=float),
                np.array(capped_y, dtype=float),
                pen=None,
                symbol="x",
                # Ink, not a hue: the marks annotate the curve rather than being a second series.
                symbolBrush=plot_theme.brush(theme.summary),
                symbolPen=None,
                symbolSize=plot_theme.MARKS.marker_size + 2,
            )

    def _count_at(self, cutoff: float) -> int:
        """The curve's instance count nearest ``cutoff`` — read, never recomputed.

        Instance count is not guaranteed to fall as the cutoff rises (§20: a method that
        re-forms extents can split one instance into two), so this reads the nearest measured
        point rather than interpolating a shape the data need not have.
        """
        if not self._curve:
            return 0
        nearest = min(self._curve, key=lambda p: abs(p["cutoff"] - cutoff))
        return int(nearest["n_instances"])

    def _refresh_view(self, cutoff: float) -> None:
        """Everything a cutoff change touches. The strip keeps its list; only its line moves."""
        self._refresh_header(cutoff)
        self._grid.set_cutoff(cutoff)

    def _refresh_header(self, cutoff: float) -> None:
        behavior = self._run.state().behavior(self._behavior_id)
        total = self._curve[0]["n_instances"] if self._curve else 0
        shown = self._count_at(cutoff)
        committed = (
            "not calibrated"
            if behavior.cutoff is None
            else f"cutoff {behavior.cutoff:.4f}"
        )
        self._header.setText(
            f"{behavior.name} ({behavior.id}) — {shown} instance(s) at {cutoff:.4f} · "
            f"{committed}"
        )
        self._confidence.set_caption(showing_caption(shown, total))
        self._draw_spread(cutoff)

    def _draw_spread(self, cutoff: float) -> None:
        """How many instances each recording keeps at this cutoff (§20).

        The panel exists to catch a cutoff that keeps most of one animal's behavior and little
        of another's — which would bias the very comparison the run is for, along the axis
        being compared, and the pooled curve cannot show it.

        It used to draw each recording's score *distribution* as block characters. That is the
        raw shape rather than the answer: users read it as "a pixelated graphic with no
        context" and could not tell what it was for. A kept count says the thing directly, and
        the outlier marker names the case the panel was built to catch.
        """
        # Seed every *searched* recording, not just the ones with candidates. A recording that
        # matched nothing has a real answer — zero — and leaving its row out reads as "not
        # searched", which is exactly the distinction coverage exists to preserve (§20).
        counts = {rid: 0 for rid in self._run.state().search_recording_ids}
        for (recording_id, _subject), candidates in self._run.candidates_at(
            self._behavior_id, 0.0
        ).items():
            counts[recording_id] = counts.get(recording_id, 0) + sum(
                1 for c in candidates if (c.confidence or 0.0) >= cutoff
            )
        if not counts:
            self._spread.setText("(no recordings searched)")
            return
        peak = max(counts.values())
        typical = sorted(counts.values())[len(counts) // 2]
        lines = []
        for recording_id, kept in sorted(counts.items()):
            bar = "█" * (int(round(20 * kept / peak)) if peak else 0)
            line = f"{recording_id:<22} {bar:<20} {kept:>4}"
            # Named rather than left to the eye: an animal contributing a fraction of the
            # others is the finding this panel is for.
            if typical and kept * 4 <= typical:
                line += "  ← far fewer than the rest"
            lines.append(line)
        self._spread.setText("\n".join(lines))

    def _refresh_clips(self) -> None:
        """Hand the strip the behavior's whole ranked list, opening at the cutoff.

        The strip is the ordering, so there is no band to choose and no near/above mode: you
        scroll to wherever you want to look.

        The list is **built once per behavior and kept**. It was described here as cheap and is
        not: it reads a `unit_timeline` off disk for every searched unit and rebuilds every spec
        from them, on the UI thread, every time a behavior is shown. None of it changes while
        the run is open — the cutoff decides what the strip *marks*, not what is in it.
        """
        cutoff = self.cutoff()
        specs = self._specs.get(self._behavior_id)
        if specs is None:
            specs = self._specs[self._behavior_id] = self._build_specs()
        self._grid.show_clips(
            specs,
            cutoff,
            empty_text=(
                f"No instances at {cutoff:.4f}. That is a result — this behavior returns "
                "nothing at this setting."
            ),
        )

    def _build_specs(self) -> list:
        """Every candidate this behavior has, as clips, in confidence order."""
        state = self._run.state()
        specs: list = []
        for (recording_id, subject), candidates in sorted(
            self._run.candidates_at(self._behavior_id, 0.0).items()
        ):
            if not candidates:
                continue
            timeline = self._timeline(recording_id, state)
            if timeline is None:
                continue
            specs.extend(
                _clip_specs(
                    candidates, recording_id, subject, state.embed_run_id, timeline
                )
            )
        return specs

    def _timeline(self, recording_id: str, state):
        """The unit's frame timeline, read once per unit and kept.

        A completed embedding run's timeline does not change, and this is called once per
        searched unit every time a behavior is shown.
        """
        from detail.pipeline.query import unit_timeline

        key = (recording_id, state.embed_run_id, state.track_id)
        if key in self._timelines:
            return self._timelines[key]
        try:
            self._timelines[key] = unit_timeline(
                self._run._experiment.dir,
                state.embed_run_id,
                recording_id,
                state.track_id,
            )
        except (FileNotFoundError, ValueError):
            # This unit's embedding is gone; its clips simply do not show. Cached too, so the
            # miss is not re-attempted on every switch.
            self._timelines[key] = None
        return self._timelines[key]

    # -- the cutoff, from either control ------------------------------------------------ #

    def _set_cutoff_widgets(self, value: float) -> None:
        self._syncing = True
        try:
            self._line.setValue(float(value))
            self._confidence.set_value(float(value))
        finally:
            self._syncing = False

    def _on_line_moved(self) -> None:
        if self._syncing:
            return
        value = self.cutoff()
        self._syncing = True
        try:
            self._confidence.set_value(value)
        finally:
            self._syncing = False
        self._refresh_view(value)

    def _on_line_released(self) -> None:
        self._commit(self.cutoff())

    def _on_confidence_typed(self, value: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._line.setValue(float(value))
        finally:
            self._syncing = False
        self._refresh_view(float(value))
        # The numeric box needs its own commit: `sigPositionChangeFinished` fires only for a
        # real drag, so moving the line programmatically above will never reach `_commit`.
        self._commit(float(value))

    def set_editable(self, editable: bool, *, reason: str) -> None:
        """Freeze the cutoff controls when the run will refuse a cutoff — a snapshot, or finished.

        There are **three** ways to move a cutoff, not two: the curve's line, the number box, and
        the strip's own ▲/▼ stepping. Freezing the first two left the third live — the write was
        still refused, but the screen showed a cutoff belonging to neither the snapshot nor the run,
        on the one screen whose purpose is "which cutoff did I ship?".

        ``reason`` is required because there are two reasons and they are undone differently:
        a snapshot by going back to the live run, a finished run by Reopen.
        """
        self._editable = editable
        self._line.setMovable(editable)
        self._confidence.setEnabled(editable)
        self._grid.set_editable(editable)
        self._confidence.setToolTip("" if editable else reason)

    def _commit(self, value: float) -> None:
        """Write the cutoff and reproject — off the UI thread, on release only."""
        if not getattr(self, "_editable", True):
            return
        behavior_id = self._behavior_id
        run = self._run

        def work(emit):
            run.set_cutoff(behavior_id, float(value), progress=emit)

        run_progress(
            self,
            work,
            lambda _r: self._on_committed(value),
            label="Applying cutoff…",
        )

    def _on_committed(self, value: float) -> None:
        self._refresh_view(value)
        # The other tabs read the artifact this just rewrote. `_BrowseTab.refresh`'s own
        # docstring said it ran "after every commit" and nothing connected it, so a committed
        # cutoff left Browse listing the previous one's instances until you switched behaviors.
        # The clustering workspace has wired `changed -> reload` all along.
        self.committed.emit()
        self._refresh_ethograms()

    def _refresh_ethograms(self) -> None:
        """Where the kept instances land, one row per searched recording.

        Read from the **artifact**, which is why this runs on commit and not on drag: the
        artifact holds only what the committed cutoff kept, so during a drag there is nothing
        to draw that corresponds to the line's current position. The title says so rather than
        leaving a reader to wonder why the bands did not move.
        """
        run_busy(
            self,
            self._ethogram_rows,
            self._show_ethograms,
            label="Reading instances…",
            supersedes="ethograms",
        )

    def _ethogram_rows(self) -> tuple:
        """Off-thread: the ethogram rows **and** where those instances happened.

        Both come off one artifact read. Two reads would be two disk trips for one answer,
        and — worse — could disagree if a commit landed between them, leaving the map
        showing a cutoff the strip beside it has already moved past.
        """
        from detail.storage import behavior_output as bo

        try:
            artifact = self._run.artifact()  # off-thread
            instances = list(artifact.instances)
            coverage = list(artifact.coverage)
            # `None` means the run never recorded positions (fixable); `{}` means it did
            # and none landed. Carried separately so the surface can say which.
            features = artifact.features
        except FileNotFoundError:
            return ([], {})
        # The span each recording covers, so a band can be a fraction of it — `EthogramStrip`
        # takes fractions precisely so it need not know how long a recording is.
        span: dict = {}
        for row in coverage:
            lo, hi = span.get(row.recording_id, (None, None))
            start, end = int(row.start_t_ns), int(row.end_t_ns)
            span[row.recording_id] = (
                start if lo is None else min(lo, start),
                end if hi is None else max(hi, end),
            )
        rows = []
        for recording_id in sorted(span):
            lo, hi = span[recording_id]
            width = max(hi - lo, 1)
            bands = [
                ((int(i.t_ns) - lo) / width, (int(i.end_t_ns) - lo) / width, 0)
                for i in bo.bouts(instances)
                if i.recording_id == recording_id
                and (self._behavior_id is None or i.source_label == self._behavior_id)
            ]
            rows.append((recording_id, bands, width))
        outlines: dict = {}
        aspects: dict = {}
        for recording_id, _bands, _width in rows:
            try:
                recording = self._experiment.get_recording(recording_id)
                # The frame's shape, off the same read: `cx` and `cy` are fractions of width and
                # height, so a box drawn square stretches a 16:9 arena and moves every dot in it.
                aspects[recording_id] = frame_aspect(recording)
                arenas = recording.arenas
                if not arenas:
                    continue  # no arena drawn: the frame *is* the field of view
                keyframes = recording.arena_keyframes(arenas[0]["id"])
            except (FileNotFoundError, ValueError, KeyError):
                continue
            if keyframes:
                outlines[recording_id] = keyframes[0][1]
        self._outlines_by_recording = outlines
        self._frame_aspects = aspects
        # Off the UI thread with the rest of the load, like the outlines above it.
        self._embed_run_id = self._run.state().embed_run_id
        if features is None:
            return (rows, None)
        # Placements, not bare coordinates. Every identity field is already in hand here, and
        # a dot that reports only where it was cannot be turned back into the clip it stands
        # for — which is the whole of clicking one.
        placements: dict = {}
        for row in bo.bouts(instances):
            if self._behavior_id is not None and row.source_label != self._behavior_id:
                continue
            found = features.get(row.instance_id)
            if found is None or found[0] is None or found[1] is None:
                continue
            placements.setdefault(row.recording_id, []).append(
                {
                    "instance_id": row.instance_id,
                    "cx": float(found[0]),
                    "cy": float(found[1]),
                    "subject": int(row.subject or 0),
                    "t_ns": int(row.t_ns),
                    "end_t_ns": int(row.end_t_ns),
                }
            )
        return (rows, placements)

    def _curve_colour(self) -> str:
        """The one colour this run's behavior is drawn in — curve, map and ethogram alike.

        A query artifact is one behavior, so it is identity 0. Named by role rather than picked
        out of a palette by index: two sites used to take ``PALETTE_OKABE_ITO[3]`` as "the same
        blue the confidence curve is drawn in" while the curve spelled ``(0, 114, 178)`` itself,
        and reordering the palette would have made them three colours without failing anything.
        """
        return plot_theme.current().identity(0)

    def _map_colour(self):
        """The curve's colour as an ``(r, g, b)`` triple, for the spatial map."""
        return chart_theme.rgb(self._curve_colour())

    def _show_ethograms(self, payload) -> None:
        rows, placements = payload
        self._placements_by_recording = placements
        kept = sum(len(bands) for _rid, bands, _width in rows)
        self._ethogram_title.setText(
            f"Kept instances, per recording — {kept} at the committed cutoff"
        )
        # `EthogramStrip` takes 0-1 floats with alpha; the shared palette is 0-255 ints. One
        # colour, because a query artifact is one behavior — the strip's cluster index is
        # always 0 here.
        r, g, b = self._map_colour()
        self._ethograms.set_rows(
            [(rid, bands) for rid, bands, _width in rows],
            [(r / 255, g / 255, b / 255, 1.0)],
            selected=None,
            lengths_ns=[width for _rid, _bands, width in rows],
        )
        # A run that never recorded positions is a fixable state and says so; a run that has
        # them where none landed here is an absence, and an empty box beside a full ethogram
        # would read as the claim "it happens nowhere".
        if placements is None:
            self._map_band.setVisible(True)
            self._map.show_unavailable(
                "This run has no recorded positions — it was made before position capture. "
                "Run `detail recording backfill-kinematics` and re-open the run to fill it in."
            )
            return
        self._map_band.setVisible(bool(placements))
        if placements:
            self._map.show_instances(
                {rid: placements.get(rid, []) for rid, _bands, _width in rows},
                outlines=self._outlines_by_recording,
                aspects=self._frame_aspects,
                embed_run_ids={rid: self._embed_run_id for rid, _bands, _width in rows},
                colour=self._map_colour(),
            )


class _BrowseTab(QWidget):
    """The accepted instance set: what you actually got, read-only (§20).

    Calibrate shows the *marginal* clips because the decision is about the edge; this shows
    everything above the cutoff, because that is where a systematic error becomes visible —
    every match in one recording being the same wall-lean is invisible in a marginal band and
    obvious in a list.
    """

    def __init__(self, run, experiment) -> None:
        super().__init__()
        from detail.gui.behavior_query import (
            _ClipPlayer,
        )  # local: avoids an import cycle

        self._run = run
        self._experiment = experiment
        self._behavior_id = None
        self._rows: list = []
        self._names: dict[str, str] = {}  # {recording_id: name}, read with the artifact

        root = QVBoxLayout(self)
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)

        # The list folds by its grip to a strip reading "Instances"; the player does not.
        splitter = GripSplitter(Qt.Orientation.Horizontal, key="query_browse")
        self._model = QStandardItemModel()
        self._table = QTableView()
        fill_width(self._table)
        self._table.setModel(self._model)
        self._table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        make_columns_resizable(self._table)
        # Sortable headers, the GUI's convention for a multi-column list. The default order is
        # by recording then time; any column can be flipped without picking one for the user.
        # `enable_sorting`, not a bare `setSortingEnabled` — the latter immediately re-sorts by
        # column 0 descending (Qt's undocumented-until-you-hit-it default indicator), which
        # would have silently replaced "by recording then time" with exactly that.
        enable_sorting(self._table)
        # Now, while it is empty, and not only once rows arrive — see `fit_table_height`.
        fit_table_height(self._table, max_rows=BROWSE_ROWS, min_rows=0)
        splitter.addWidget(top_of_pane(self._table))
        # The clip and nothing either side — see Calibrate's player. Browse watches accepted
        # instances and never edits their bounds, so it has no use for the padding either.
        # (`context_s=0` is not a whole-recording scrubber: it spans exactly the clip.)
        # At the top, beside the list's first rows — see the cluster workspace's Browse.
        self._player = _ClipPlayer(experiment, autoplay=True, context_s=0, top=True)
        splitter.addWidget(self._player)
        splitter.set_foldable(0, "Instances")
        if not splitter.remember():
            splitter.setSizes([420, 460])
        root.addWidget(splitter, 1)

        self._table.selectionModel().currentRowChanged.connect(self._show_row)

    def set_behavior(self, behavior_id: str) -> None:
        self._behavior_id = behavior_id
        self.refresh()

    def refresh(self) -> None:
        """Re-read the artifact and rebuild the table.

        The read is off the UI thread: ``instances.csv`` grows with the cohort, and this runs
        on every tab switch and after every commit.
        """
        state = self._run.state()
        behavior = state.behavior(self._behavior_id)
        # Cached here, once per refresh, rather than read again for every row someone clicks.
        self._embed_run_id = state.embed_run_id
        self._model.clear()
        self._model.setHorizontalHeaderLabels(
            ["Recording", "Start", "End", "Confidence"]
        )
        self._rows = []
        fit_table_height(self._table, max_rows=BROWSE_ROWS, min_rows=0)
        if behavior.cutoff is None:
            self._summary.setText(
                f"{behavior.name} has no cutoff yet, so it has no instances. Set one on "
                "Calibrate."
            )
            return
        run = self._run
        experiment = self._experiment

        def read():
            # The names come with the artifact: one read per recording, off the UI thread.
            try:
                return run.artifact(), experiment.recording_names()  # off-thread
            except FileNotFoundError:
                return None, {}

        run_busy(
            self, read, self._fill, label="Loading instances…", supersedes="instances"
        )

    def _name(self, recording_id: str) -> str:
        """The recording's name, or its id if it has none left (deleted since the run)."""
        return self._names.get(recording_id, recording_id)

    def _fill(self, payload) -> None:
        output, self._names = payload
        behavior = self._run.state().behavior(self._behavior_id)
        if output is None:
            self._summary.setText("(this run has no behavior output yet)")
            return
        rows = [r for r in output.instances if r.source_label == behavior.id]
        digits = score_digits([r.score for r in rows])
        for row in sorted(rows, key=lambda r: (r.recording_id, r.t_ns)):
            index = len(self._rows)
            self._rows.append(row)
            # Sorting by **Recording** reproduces the default order exactly, because that
            # column's key carries the time as its tiebreak. Chronology within an animal is
            # what makes a systematic error visible — every match being the same wall-lean —
            # so restoring it has to be one click, not a reload.
            name = self._name(row.recording_id)
            first = SortCell(name, (name, int(row.t_ns)))
            # The view reorders the model, so a row's identity cannot be its position.
            first.setData(index, Qt.ItemDataRole.UserRole)
            self._model.appendRow(
                [
                    first,
                    SortCell(f"{row.t_ns / 1e9:.2f} s", int(row.t_ns)),
                    SortCell(f"{row.end_t_ns / 1e9:.2f} s", int(row.end_t_ns)),
                    SortCell(
                        "—" if row.score is None else f"{row.score:.{digits}f}",
                        -1.0 if row.score is None else float(row.score),
                    ),
                ]
            )
        # No floor: the list sits beside the player in a splitter, which decides the share.
        fit_table_height(self._table, max_rows=BROWSE_ROWS, min_rows=0)
        covered = len(output.coverage)
        self._summary.setText(
            f"{behavior.name} at cutoff {behavior.cutoff:.2f}: {len(rows)} instance(s) "
            f"over {covered} searched unit(s)."
            + ("  This behavior returns nothing at this cutoff." if not rows else "")
        )

    def export_scope(self):
        """The instances picked in the list, for the shell's Export control.

        The row's identity travels in UserRole, not in its position: the view re-sorts the
        model, so a row number stops meaning what it meant.
        """
        from detail.api.exporters import ExportScope

        picked = []
        for index in self._table.selectionModel().selectedRows():
            item = self._model.item(index.row(), 0)
            row_index = None if item is None else item.data(Qt.ItemDataRole.UserRole)
            if row_index is not None and 0 <= int(row_index) < len(self._rows):
                picked.append(self._rows[int(row_index)].instance_id)
        return ExportScope(instance_ids=tuple(picked)) if picked else None

    def _show_row(self, current, _previous) -> None:
        """Play the selected instance.

        `embed_run_id` comes off the cached load rather than `state()`: that is a disk read, and
        a row someone clicks must not pay for one on the UI thread.
        """
        from detail.gui.behavior_query import ClipSpec

        if not current.isValid():
            return
        index = self._model.item(current.row(), 0).data(Qt.ItemDataRole.UserRole)
        if index is None or int(index) >= len(self._rows):
            return
        row = self._rows[int(index)]
        self._player.set_clip(
            ClipSpec(
                row.recording_id,
                int(row.subject or 0),
                self._embed_run_id,
                int(row.t_ns),
                int(row.end_t_ns),
                f"{self._name(row.recording_id)} {row.t_ns / 1e9:.2f}s",
                score=row.score,
            )
        )


#: The behavior-scan workspace's views, in row order: ``(name, purpose)``. Measures is live-only.
QUERY_VIEWS = (
    ("Calibrate", "Set each behavior's cutoff against how often it would be found."),
    ("Audit", "Judge clips above the cutoff, to see how often a find is right."),
    ("Browse", "Every find above the cutoff, playing one at a time."),
    ("Measures", "Time, bouts and transitions per behavior."),
)


def _behaviors_status(run) -> str:
    count = len(run.behaviors())
    return f"{count} behavior{'' if count == 1 else 's'}"


class _QueryWorkspace(MultiViewWorkspace):
    """A behavior selector over Calibrate and Browse (§20).

    Switching behavior keeps the current view, so the selector sits outside the tab widget and
    each view is re-pointed rather than rebuilt. The views are full-window ones the run page
    switches between (`MultiViewWorkspace`); the selector stays inside each of them.

    The search results load **off the UI thread**: reading a shard per searched unit is
    O(recordings) disk, and this widget is built inline by the run-detail page, so doing it in
    ``__init__`` would freeze the page for as long as the load takes.
    """

    def __init__(self, experiment, run) -> None:
        """``run`` is the handle, not an id.

        Handed one rather than building it, so the same workspace renders a **snapshot view** — a
        frozen handle that reads the calibration as it was pinned and refuses every write (§21).
        Constructing the handle here is what kept this surface from doing that.
        """
        super().__init__()
        self._run = run
        self._experiment = experiment
        self._curves: dict[str, list] = {}

        root = QVBoxLayout(self)
        self._status = QLabel("Loading search results…")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._selector = QTabBar()
        self._behaviors = list(self._run.behaviors())
        for behavior in self._behaviors:
            self._selector.addTab(behavior.name)
        self._selector.currentChanged.connect(self._on_behavior_changed)
        root.addWidget(self._selector)

        self._tabs = QTabWidget()
        self._calibrate = _CalibrateTab(self._run, experiment)
        self._audit = AuditTab(experiment, self._run)
        self._browse = _BrowseTab(self._run, experiment)
        # The tab is a leaf and does not know this workspace, so it asks rather than reaching
        # for the selector; driving the selector keeps the two in sync in both directions.
        self._audit.behavior_requested.connect(self._select_behavior)
        # A commit rewrites the artifact the other two tabs read.
        self._calibrate.committed.connect(self._browse.refresh)
        self._calibrate.committed.connect(self._audit.reload)
        self._tabs.addTab(self._calibrate, "Calibrate")
        self._tabs.addTab(self._audit, "Audit")
        self._tabs.addTab(self._browse, "Browse")
        # The §23 numbers as a tab, not a section under every tab (see `ClusterWorkspace`).
        # Live runs only: they read the live artifact.
        if self._run.editable:
            from detail.gui.pages import (
                MeasuresPanel,
            )  # local: pages imports this module

            self._tabs.addTab(
                top_of_pane(MeasuresPanel(experiment, self._run.id)), "Measures"
            )
        statuses = (lambda: _behaviors_status(self._run), None, None, None)
        self._init_views(
            self._tabs,
            [
                (name, purpose, self._tabs.widget(index), statuses[index])
                for index, (name, purpose) in enumerate(
                    QUERY_VIEWS[: self._tabs.count()]
                )
            ],
        )
        root.addWidget(self._tabs, 1)

        actions = QHBoxLayout()
        self._finish = QPushButton("Finish")
        self._finish.clicked.connect(self._on_finish)
        self._reopen = QPushButton("Reopen")
        self._reopen.clicked.connect(self._on_reopen)
        actions.addWidget(self._finish)
        actions.addWidget(self._reopen)
        actions.addStretch()
        root.addLayout(actions)

        self._sync_status()
        if self._behaviors:
            self._load_behavior(self._behaviors[0].id)

    def export_scope(self):
        """What the shell's Export control should offer — asked of the view in front.

        Delegated rather than searched, for the reason `ClusterWorkspace.export_scope` is: the
        tabs carry different kinds of selection, and answering with whichever was found first
        would export the tab the user is not looking at.
        """
        current = self._tabs.currentWidget()
        getter = getattr(current, "export_scope", None)
        return getter() if getter is not None else None

    def _load_behavior(self, behavior_id: str) -> None:
        if behavior_id in self._curves:
            self._show_behavior(behavior_id)
            return
        run = self._run

        def work(emit):
            # `run_progress` runs this on a worker. Loading is the sized step — a shard per
            # searched unit, plus the extraction each one needs — and the curve that follows
            # is a slice of it, so both belong on the same side of the thread boundary.
            run.load_results(behavior_id, progress=emit)  # off-thread
            return run.frequency_curve(behavior_id, steps=_CURVE_STEPS)

        run_progress(
            self,
            work,
            lambda curve: self._on_loaded(behavior_id, curve),
            label="Loading search results…",
            # Switching behavior mid-load: the older one landing last showed it, not the choice.
            supersedes="results",
        )

    def _on_loaded(self, behavior_id: str, curve) -> None:
        self._curves[behavior_id] = list(curve)
        self._show_behavior(behavior_id)

    def _show_behavior(self, behavior_id: str) -> None:
        self._calibrate.set_behavior(behavior_id, self._curves.get(behavior_id, []))
        self._audit.set_behavior(behavior_id)
        self._browse.set_behavior(behavior_id)
        self._sync_status()

    def _select_behavior(self, behavior_id: str) -> None:
        """Move the selector to a behavior the audit table asked for."""
        for index, behavior in enumerate(self._behaviors):
            if behavior.id == behavior_id:
                self._selector.setCurrentIndex(index)
                return

    def _on_behavior_changed(self, index: int) -> None:
        if 0 <= index < len(self._behaviors):
            self._load_behavior(self._behaviors[index].id)

    def _sync_status(self) -> None:
        status = self._run.status
        editable = self._run.editable
        self._status.setText(
            f"Run {self._run.id} is {status}."
            if editable
            else f"Reading {self._run.snapshot_id} — the calibration as it was pinned. Read-only."
        )
        # Finish is a statement that you are done, not a lock — Browse still works afterwards
        # and reopening is one click. So the pair is enabled by which one applies, not hidden.
        #
        # A snapshot is the *other* reason a control is dead, and not the same one: finished is
        # undone by Reopen, frozen by going back to the live run. Saying which is the difference
        # between a user knowing what to do and guessing.
        frozen = (
            "This is a snapshot. Go back to the live run to change the calibration."
        )
        finished = status == "complete"
        self._finish.setEnabled(editable and not finished)
        self._reopen.setEnabled(editable and finished)
        for button in (self._finish, self._reopen):
            button.setToolTip("" if editable else frozen)
        # Finished as well as frozen: `set_cutoff` refuses both, and freezing only the snapshot
        # left a finished run's cutoff draggable into an error.
        self._calibrate.set_editable(
            editable and not finished,
            reason=(
                "Fixed by this snapshot. Go back to the live run to change it."
                if not editable
                else "This run is finished. Reopen it to change a cutoff."
            ),
        )

    def _after_status_change(self) -> None:
        self._sync_status()
        # The audit asks the run's status on its own read, so it is told to read again.
        self._audit.reload()

    def _on_finish(self) -> None:
        # Off the UI thread: finishing pins, and a pin first rebuilds the frame views the cutoffs
        # since the last one deferred.
        run_progress(
            self,
            lambda emit: self._run.finalize(progress=emit),
            lambda _result: self._after_status_change(),
            label="Finishing…",
        )

    def _on_reopen(self) -> None:
        self._run.reopen()
        self._after_status_change()


@register_analysis_module_ui
class QueryModuleUI(ModuleUI):
    module_id = "query"
    # The user-facing name. The module id, its CLI command and its on-disk records stay
    # `query` — renaming those would orphan every run already written.
    # One writer for one fact: the lifecycle flag is the backend module's (§21 moves it onto
    # the AnalysisModule so the API and CLI know it too), and this reads it rather than
    # restating it.
    mutable = QueryModule.mutable

    def config_widget(self) -> ConfigWidget:
        return _QueryConfig()

    def planned_views(self) -> list[tuple[str, str]]:
        return list(QUERY_VIEWS)

    def result_widget(self, experiment, run_record: dict) -> QWidget:
        """The calibration workspace — or why there is not one yet.

        A run reaches here while still ``open``: §21 says an authored run's statistics are what
        the user is deciding against, so this surface exists before the run is finished. What
        it cannot do is calibrate a run whose search never wrote its state, so that case says
        so rather than opening an empty workspace onto nothing.
        """
        from detail.storage import query_runs

        run_dir = experiment.analysis_run_dir(run_record.get("id", ""))
        if not query_runs.has_query_state(run_dir):
            return _placeholder("(this run has no query state)")
        removed = _removed_method(run_dir)
        if removed is not None:
            return _placeholder(removed)
        return _QueryWorkspace(
            experiment, experiment.query_run(run_record.get("id", ""))
        )

    def snapshot_widget(
        self, experiment, run_record: dict, snapshot_id: str
    ) -> QWidget:
        """The same workspace, driven by the calibration as it was pinned (§21).

        The cutoffs are the point: a snapshot answers "which cutoff produced the number I quoted",
        which the run itself stops answering the moment somebody drags the line again.
        """
        removed = _removed_method(experiment.analysis_run_dir(run_record.get("id", "")))
        if removed is not None:
            return _placeholder(removed)
        run = experiment.query_run(run_record.get("id", ""))
        return _QueryWorkspace(experiment, run.view(snapshot_id))


def _placeholder(text: str) -> QWidget:
    """A result pane that says why there is no workspace, instead of an empty one."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    label = QLabel(text)
    label.setWordWrap(True)
    layout.addWidget(label)
    layout.addStretch()
    return widget


def _removed_method(run_dir) -> str | None:
    """Why a saved run's workspace cannot open, when its method is no longer in this build.

    The workspace reads the stored search through the method that wrote it, so a run whose
    method was removed (``docs/removed_capabilities.md``) would raise from a worker thread and
    leave the pane loading forever. The record, its artifact and its output folder are kept.
    """
    from detail.core.registries import QUERY_METHODS
    from detail.query_methods import import_query_methods
    from detail.storage import query_runs

    import_query_methods()
    method = query_runs.read_query_state(run_dir).method
    if method in QUERY_METHODS:
        return None
    return (
        f"This run was searched with the '{method}' method, which this version no longer "
        "has, so its search cannot be calibrated or reviewed here. The run record, its "
        "behavior output and its output folder are kept."
    )
