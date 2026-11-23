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
authored run (§21) and which the removed ``clip_annotation`` established. Audit is
``query_review.QueryReviewTab``: every clip a behavior's cutoff keeps, reviewed one recording at a
time on the clustering review's pane (§20 *The audit*).
"""

from detail.configs.query import DEFAULT_METHOD, QueryConfig
from detail.core import chart_theme
from detail.gui.forms import ConfigForm
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
from detail.gui.clip_grid import score_digits
from detail.gui.confidence import ConfidenceRow, showing_caption
from detail.gui.collapse import CollapsibleBand
from detail.gui.live_montage import LiveMontage, MontageClip, context_span
from detail.gui.measurements import BASELINE_RECORDING, MeasurementsPanel
from detail.gui.cluster_workspace import QUERY_TERMS
from detail.gui.label_browse import LabelBrowseTab
from detail.gui.qt import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QEvent,
    QFont,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPalette,
    QPushButton,
    QScrollArea,
    QStandardItem,
    QStandardItemModel,
    QTabWidget,
    QTableView,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)
from detail.gui.query_review import QueryReviewTab
from detail.gui.tables import (
    num_cell,
    enable_sorting,
    make_columns_resizable,
    row_id,
    stamp_row_id,
)
from detail.gui.sizing import emphasis_font, lines, set_available
from detail.gui.timefmt import format_time
from detail.gui.spatial_map import SpatialMap, frame_aspect
from detail.gui.method_form import MethodFields
from detail.gui.example_spread import spread_text
from detail.gui import plot_theme
from detail.gui.plots import bound_axes
from detail.gui.sizing import lines
from detail.gui.tables import fill_width
from detail.pipeline.query import QueryModule, resolve_method
from detail.storage import query_runs
from detail.gui.status_color import (
    MUTED,
    WARN,
    set_band,
    set_label_color,
    warn_ink,
)
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
    noted_by_form = frozenset({"method"})  # its own note, beside the row

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
        self._embed.currentIndexChanged.connect(
            lambda _index: self.coverage_changed.emit()
        )
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


# Points on the frequency curve. Enough that the shape is legible, few enough that building it
# stays one pass over the cached extraction.
_CURVE_STEPS = 41
#: The curve's height floor, in lines of the interface font (`sizing.lines`), so a user who
#: scales their text gets a taller graph rather than the same box with bigger axis labels.
#:
#: The floor is what a curve is still readable at. pyqtgraph's axis chrome is a *constant* ~40px
#: whatever the widget's height, so a 130px plot still spends 90px on data — measured across
#: 240/160/120/90px, the data area is 200/120/80/50. Below about eight lines the chrome is most of
#: it and the shape stops being legible. Its ceiling is the band's (`_BAND_MAX_LINES`).
#: A line *count*, resolved at construction: there is no application at import time.
_PLOT_MIN_LINES = 8


def _sparkline_font() -> QFont:
    """Monospace, so the per-recording bars line up into a readable column."""
    return mono_font()


def _clip_specs(
    candidates,
    recording_id,
    subject,
    embed_run_id,
    timeline,
    *,
    name: str,
    removed: frozenset = frozenset(),
) -> list:
    """``QueryCandidate``s -> ``ClipSpec``s, which every clip surface in the GUI speaks.

    ``name`` is the recording's, for the caption (§11: the id is the tool's). Required, so a
    caller cannot fall back to the id by forgetting it. A clip whose span is in ``removed`` (the
    behavior's review took it out, §20) says so in its caption: the strip is the method's own
    ranked list, so a removed clip stays in it, but it is not in the behavior's instances.

    The candidate carries trajectory **rows**; a clip surface needs times, so the unit's
    timeline supplies them by direct indexing — the same projection the artifact writer does.
    """
    from detail.gui.behavior_query import ClipSpec

    _frame_idx, t_ns = timeline
    out = []
    for c in candidates:
        if c.start_row >= len(t_ns) or c.end_row >= len(t_ns):
            continue  # a shard from a search whose trajectory has since changed
        start, end = int(t_ns[c.start_row]), int(t_ns[c.end_row])
        mark = (
            " · removed" if (recording_id, int(subject), start, end) in removed else ""
        )
        out.append(
            ClipSpec(
                recording_id,
                int(subject),
                embed_run_id,
                start,
                end,
                f"{name} {c.confidence:.2f}{mark}",
                score=float(c.confidence),
            )
        )
    return out


#: Where Calibrate's grid keeps its orientation, layers and context — under ``query/``, apart
#: from AUDIT's (``query_review``) and clustering's.
CALIBRATE_ALIGNED_KEY = "query/calibrate_aligned"
CALIBRATE_LOOK_KEY = "query/calibrate_look"
CALIBRATE_CONTEXT_KEY = "query/calibrate_context"
#: The band over the grid — the curve beside the cutoff controls — as a height range in lines.
#: Its floor is the curve's legibility floor plus a line; its ceiling keeps the grid the page's
#: largest thing, which is the point of the layout (the clips are what the cutoff is judged on).
_BAND_MAX_LINES = 14
#: The cutoff controls' width beside the curve.
_CONTROLS_PX = 440
#: A tile loops at least this long, so a one-frame match is still watchable.
_MIN_LOOP_NS = 400_000_000


def _cutoff_text(value: float) -> str:
    """A cutoff as a person reads it: up to four decimals, without trailing zeros."""
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text or "0"


def not_saved_text(name: str, shown: float, saved: float | None, keeps: str) -> str:
    """CALIBRATE's *not saved* bar: what is shown, and what every other view still uses."""
    here = _cutoff_text(shown)
    if saved is None:
        return (
            f"⚠ Not saved. You're looking at {here}, which would keep {keeps}; {name} has no "
            "cutoff yet, so Audit, Browse and Measures have nothing to show."
        )
    return (
        f"⚠ Not saved. You're looking at {here}, which would keep {keeps}; Audit, Browse and "
        f"Measures still use {_cutoff_text(saved)}."
    )


def unsaved_notice_text(name: str, shown: float, saved: float | None) -> str:
    """The notice the other views show while CALIBRATE holds an unsaved cutoff."""
    here = _cutoff_text(shown)
    if saved is None:
        return (
            f"⚠ Calibrate shows {name} at {here}, not saved, and {name} has no cutoff yet: "
            "this view has nothing from it."
        )
    return (
        f"⚠ Calibrate shows {name} at {here}, not saved. This view uses the saved "
        f"{_cutoff_text(saved)}."
    )


class _AttentionBar(QFrame):
    """A band that is plainly not the page (`status_color.set_band`), words in amber and bold,
    and the buttons that resolve it. Shared by CALIBRATE's *not saved* bar and the other views'
    notice, so the two look and read as one thing."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        row = self._row = QHBoxLayout(self)
        self.text = QLabel("")
        self.text.setWordWrap(True)
        row.addWidget(self.text, 1)
        self.set_loud(True)

    def add_button(self, button: QPushButton) -> None:
        self._row.addWidget(button)

    def set_loud(self, loud: bool) -> None:
        """Loud: banded, amber, bold — something is not saved. Quiet: the page's own line."""
        self.setFrameShape(QFrame.Shape.Box if loud else QFrame.Shape.NoFrame)
        set_band(self, loud)
        font = self.text.font()
        font.setBold(loud)
        self.text.setFont(font)
        if loud:
            set_label_color(self.text, warn_ink(self))
        else:
            self.text.setPalette(self.palette())


def _clock(t_ns: int) -> str:
    """``m:ss`` of a recording time, for a tile's caption."""
    seconds = int(t_ns) // 1_000_000_000
    return f"{seconds // 60}:{seconds % 60:02d}"


class _BelowCutoffBar(QFrame):
    """The bar pinned over CALIBRATE's grid while everything in view is below the cutoff."""

    #: **↑ Back to the cutoff** was pressed.
    back_requested = Signal()

    def __init__(self, parent) -> None:
        super().__init__(parent)
        # A band that is plainly not the page (`set_band`), framed, its words bold and amber: the
        # first build used the page's own colour and vanished over the tiles in a dark theme.
        self.setFrameShape(QFrame.Shape.Box)
        set_band(self)
        row = QHBoxLayout(self)
        row.setContentsMargins(lines(0.5), 4, lines(0.5), 4)
        self._text = QLabel("")
        self._text.setFont(emphasis_font(self._text, steps=0))
        set_label_color(self._text, warn_ink(self))
        self.back = QPushButton("↑ Back to the cutoff")
        self.back.setObjectName("calibrate_back_to_cutoff")
        self.back.clicked.connect(self.back_requested.emit)
        # Centred together, the button beside the words it acts on.
        row.addStretch(1)
        row.addWidget(self._text)
        row.addSpacing(lines(1))
        row.addWidget(self.back)
        row.addStretch(1)

    def set_text(self, text: str, *, back: bool) -> None:
        self._text.setText(text)
        self.back.setVisible(back)

    def text(self) -> str:
        return self._text.text()


class _CalibrateTab(QWidget):
    """Where a behavior's cutoff is chosen: the curve and its controls over every clip, playing.

    Laid out like clustering's INTERPRET — the map across the top (there the tree, here the
    frequency curve) and every clip playing in a grid below it (``LiveMontage``), most confident
    first. The cutoff is a dashed rule across that grid (``LiveMontage.set_rule_headings``),
    labelled "▲ N kept · cutoff X · M dropped ▼", with the dropped clips starting on the row
    after it; nothing is faded. Scrolled past it, a bar pinned over the grid says so and goes
    back. Clicking a clip moves the cutoff to it and keeps that clip where it was on screen; a
    double-click opens it in the clip window. A clip the search found on top of the behavior's own
    example or non-example is tagged so (``query_runs.told_kind``).

    **Moving the cutoff previews; one button saves it** (§20). The line, the slider, the number
    and a clicked clip all move the *shown* cutoff, which re-sections the grid and recounts — a
    slice of results already loaded. **Keep clips from X up** is the only write: it saves the
    cutoff and reprojects the artifact, off the UI thread. A cutoff that saved itself on release
    left the page reading "not calibrated" under a slider sitting at 0.50, and the one button that
    looked like the save (*Cut here*) was grey until a clip was clicked — so a user could not
    tell how to set the thing AUDIT asked for.
    """

    #: A cutoff was written and the artifact reprojected — the other tabs are stale.
    committed = Signal()
    #: The cutoff on screen, or the saved one, changed: whether it is saved may have too.
    shown_changed = Signal()

    def __init__(self, run, experiment) -> None:
        super().__init__()
        import pyqtgraph as pg

        from detail.gui.montage_export import ExportSource

        self._run = run
        self._experiment = experiment
        self._behavior_id = None
        self._editable = True
        self._read_only_reason = ""
        #: Built clip lists (most confident first) and unit timelines, per behavior / per unit.
        #: Neither changes while the run is open; the cutoff decides what the grid *marks*.
        self._specs: dict[str, list] = {}
        self._timelines: dict[tuple, object] = {}
        self._curve: list = []
        #: ``{recording_id: name}``, read off the UI thread with the results (`set_names`).
        self._names: dict[str, str] = {}
        #: The preview cache's readers, opened once per page (``clip_tiles.TileCatalog``), and the
        #: recordings that have no preview here yet — read off the UI thread on first show.
        self._catalog = None
        self._missing: list[str] = []
        self._tiles_read = False
        #: Which behavior's clips the grid's tile source was built for.
        self._source_for: str | None = None
        self._syncing = False  # several controls drive one cutoff; do not let them echo

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # -- the line that says where you are, and the one write -------------------------- #
        # Loud — banded, amber, bold — while the cutoff on screen is not the saved one: a
        # previewed cutoff read as applied, and AUDIT then reviewed the saved one.
        self._save_bar = _AttentionBar()
        self._header = self._save_bar.text
        self._keep = QPushButton("Keep clips from 0.5 up")
        self._keep.setDefault(True)
        self._keep.clicked.connect(self._on_keep)
        self._back = QPushButton("Back to the saved cutoff")
        self._back.setObjectName("calibrate_back_to_saved")
        self._back.clicked.connect(self._back_to_saved)
        self._save_bar.add_button(self._keep)
        self._save_bar.add_button(self._back)
        outer.addWidget(self._save_bar)

        # -- the band: the curve beside its controls -------------------------------------- #
        band = self._band = QWidget()
        row = QHBoxLayout(band)
        row.setContentsMargins(0, 0, 0, 0)
        self._plot = plot_theme.themed_plot()
        # A range, not a fixed height (see `_PLOT_MIN_LINES`): a fixed height crops the axis on
        # a short window rather than letting the band give way.
        self._plot.setMinimumHeight(lines(_PLOT_MIN_LINES))
        self._plot.setLabel("bottom", "confidence cutoff")
        self._plot.setLabel("left", "instances kept")
        self._plot.setToolTip(
            "Each point is how many instances that cutoff returns. Left is loose, right is "
            "strict. ✕ marks a point where the method stopped storing candidates — the curve "
            "flattens there for that reason, not the animal's."
        )
        bound_axes(self._plot, x=(0.0, 1.0), y=(0.0, None))
        theme = plot_theme.current()
        self._line = pg.InfiniteLine(
            pos=0.5,
            angle=90,
            movable=True,
            pen=plot_theme.pen(theme.handle, width=2),
            hoverPen=plot_theme.pen(theme.handle, width=4),
        )
        # `api.query.set_cutoff` refuses a value outside [0, 1]; this stops a drag reaching it.
        self._line.setBounds([0.0, 1.0])
        # A drag previews. Nothing here writes: the release used to, and that is the save this
        # page now asks for by name (the class docstring).
        self._line.sigPositionChanged.connect(self._on_line_moved)
        self._plot.addItem(self._line)
        row.addWidget(self._plot, 1)

        controls = QWidget()
        column = QVBoxLayout(controls)
        column.setContentsMargins(0, 0, 0, 0)
        self._confidence = ConfidenceRow(
            heading="Cutoff",
            help_text=(
                "How confident a match must be to count as an instance. There is no ground "
                "truth to optimize against, so this is chosen by looking at what each "
                "setting returns — the curve beside it and the clips below. Moving it only "
                "shows what it would keep; Keep saves it."
            ),
        )
        self._confidence.threshold_changed.connect(self._on_confidence_typed)
        column.addWidget(self._confidence)
        column.addWidget(QLabel("Kept per recording, at this cutoff"))
        self._spread = QLabel("")
        self._spread.setFont(_sparkline_font())
        self._spread.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        # Its own scroll: a cohort of many recordings must not push the band into the grid.
        spread_scroll = QScrollArea()
        spread_scroll.setWidgetResizable(True)
        spread_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        spread_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        spread_scroll.setWidget(self._spread)
        column.addWidget(spread_scroll, 1)
        controls.setFixedWidth(_CONTROLS_PX)
        row.addWidget(controls)
        band.setMaximumHeight(lines(_BAND_MAX_LINES))
        outer.addWidget(band)

        # -- every clip, playing ---------------------------------------------------------- #
        self._grid_note = QLabel("")
        self._grid_note.setWordWrap(True)
        self._grid_note.setVisible(False)
        outer.addWidget(self._grid_note)
        self._montage = LiveMontage(
            aligned_key=CALIBRATE_ALIGNED_KEY,
            look_key=CALIBRATE_LOOK_KEY,
            context_key=CALIBRATE_CONTEXT_KEY,
            export=ExportSource(
                clips=self.export_clips,
                experiment_dir=lambda: self._experiment.dir,
                origin=lambda: None,
                name=self._export_stem,
                empty=lambda: "No clip is kept at this cutoff",
            ),
        )
        # Confidence is the order, so the montage's own order switch has nothing to offer.
        self._montage.order_button.setVisible(False)
        self._montage.setMinimumHeight(self._montage.height_for_rows(1))
        self._montage.set_click_hint(self._click_hint, runs=False)
        self._montage.set_rule_headings(True)
        self._montage.clip_clicked.connect(self._on_tile_clicked)
        # The caption is short; the tooltip has the whole score and the recording's whole name.
        self._montage.set_hover_text(self._hover_text)
        self._montage.clip_activated.connect(
            lambda i: QTimer.singleShot(0, lambda: self._play_clip(i))
        )
        self._montage.context_changed.connect(
            lambda _s: self._show_tiles(new_source=True)
        )
        outer.addWidget(self._montage, 1)
        #: How many clips the shown cutoff keeps — the line sits after them.
        self._kept_count = 0
        # Over the grid's top edge, not above it in the layout: appearing must not push the
        # tiles down under the pointer.
        self._below_bar = _BelowCutoffBar(self._montage.grid.viewport())
        self._below_bar.back_requested.connect(
            lambda: self._scroll_to_cutoff(self._kept_count)
        )
        self._below_bar.hide()
        self._montage.grid.verticalScrollBar().valueChanged.connect(
            self._sync_below_bar
        )
        self._montage.grid.viewport().installEventFilter(self)

    # -- what the workspace drives ---------------------------------------------------- #
    def set_behavior(self, behavior_id: str, curve: list) -> None:
        """Point the tab at a behavior whose results are already warmed."""
        self._behavior_id = behavior_id
        self._curve = list(curve)
        behavior = self._run.state().behavior(behavior_id)
        start = 0.5 if behavior.cutoff is None else float(behavior.cutoff)
        self._draw_curve()
        self._set_cutoff_widgets(start)
        self._refresh_header(start)
        if self._tiles_read:
            self._show_tiles(new_source=True, scroll=True)
        else:
            self._read_tiles()

    def cutoff(self) -> float:
        """The cutoff the page shows — saved or not."""
        return float(self._line.value())

    def set_editable(self, editable: bool, *, reason: str) -> None:
        """Freeze every way of moving the cutoff when the run will refuse one.

        There are four: the line, the number, a clicked clip, and Keep. ``reason`` is required:
        a snapshot is undone by going back to the live run, a finished run by Reopen.
        """
        self._editable = editable
        self._read_only_reason = "" if editable else reason
        self._line.setMovable(editable)
        self._confidence.setEnabled(editable)
        self._confidence.setToolTip("" if editable else reason)
        self._montage.set_click_hint(self._click_hint if editable else None, runs=False)
        if self._behavior_id is not None:
            self._refresh_header(self.cutoff())

    def set_names(self, names: dict[str, str]) -> None:
        """The recordings' names, read off the UI thread by the workspace's load."""
        if names != self._names:
            self._names = dict(names)
            self._specs.clear()  # their captions carry the name

    def forget_clips(self) -> None:
        """The review removed or put back a clip: rebuild the tiles' marks and the count."""
        self._specs.clear()
        if self._behavior_id is not None:
            self._show_tiles()
            self._refresh_header(self.cutoff())

    def _name(self, recording_id: str) -> str:
        """The recording's name, or its id if it has none left (deleted since the run)."""
        return self._names.get(recording_id, recording_id)

    # -- the curve -------------------------------------------------------------------- #
    def _draw_curve(self) -> None:
        import numpy as np

        self._plot.clear()
        self._plot.addItem(self._line)
        if not self._curve:
            return
        x = np.array([p["cutoff"] for p in self._curve], dtype=float)
        y = np.array([p["n_instances"] for p in self._curve], dtype=float)
        # Both axes pinned: with auto-range on, dragging the line nudged the x range — the graph
        # moving under the hand that moved it. The domain is [0, 1], known before the data.
        self._plot.setXRange(0.0, 1.0, padding=0.02)
        self._plot.setYRange(0.0, float(max(y.max(), 1.0)), padding=0.05)
        self._plot.disableAutoRange()
        theme = plot_theme.current()
        self._plot.plot(x, y, pen=plot_theme.pen(self._curve_colour()))
        # A curve that flattens because the method stopped storing candidates looks exactly
        # like one that flattens because the animal stopped, so those points are marked (§20).
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

    def _curve_colour(self) -> str:
        """The one colour this run's behavior is drawn in. A query artifact is one behavior, so
        it is identity 0, named by role rather than picked out of a palette by index."""
        return plot_theme.current().identity(0)

    # -- counts ----------------------------------------------------------------------- #
    def _kept_at(self, cutoff: float) -> dict[str, int]:
        """``{recording_id: instances}`` saving ``cutoff`` would keep, every searched recording
        included (one that matched nothing keeps 0, not absence).

        Exact, through the method's own ``instances_at`` (``candidates_at``) — the call the save
        projects through — not read off the curve's nearest point (#35). A slice of the results
        the workspace already loaded, cheap enough per drag.
        """
        kept = {rid: 0 for rid in self._run.state().search_recording_ids}
        for (recording_id, _subject), candidates in self._run.candidates_at(
            self._behavior_id, cutoff
        ).items():
            kept[recording_id] = kept.get(recording_id, 0) + len(candidates)
        return kept

    def _refresh_header(self, cutoff: float) -> None:
        """Where the cutoff stands — not set, saved, or shown and not saved — and the count."""
        behavior = self._run.state().behavior(self._behavior_id)
        total = self._curve[0]["n_instances"] if self._curve else 0
        kept = self._kept_at(cutoff)
        shown = sum(kept.values())
        removed = self._removed_at(cutoff, behavior)
        gone = f", {removed:,} of them removed in Audit" if removed else ""
        here = _cutoff_text(cutoff)
        count = f"{shown:,} of {total:,} clips{gone}"
        saved = behavior.cutoff
        is_saved = saved is not None and abs(float(saved) - cutoff) < 1e-9
        # The symbol and the words carry the state as well as the band does (a colour-blind
        # reader), and the band is loud only while something is not saved.
        if is_saved:
            text = f"✓ {behavior.name}: cutoff {here} saved — keeps {count}"
        else:
            text = not_saved_text(behavior.name, cutoff, saved, count)
        self._header.setText(text)
        self._save_bar.set_loud(not is_saved)
        # Dashed on the curve while not saved, solid once it is: the line says it too.
        theme = plot_theme.current()
        self._line.setPen(plot_theme.pen(theme.handle, width=2, dashed=not is_saved))
        self._back.setText(
            "Back to the saved cutoff"
            if saved is None
            else f"Back to {_cutoff_text(saved)}"
        )
        self._back.setVisible(saved is not None and not is_saved)
        set_available(
            self._back,
            self._editable and saved is not None and not is_saved,
            self._read_only_reason or "This is the saved cutoff.",
        )
        self._keep.setText(f"Keep clips from {here} up")
        if not self._editable:
            set_available(self._keep, False, self._read_only_reason)
        elif is_saved:
            set_available(self._keep, False, "This is the saved cutoff.")
        else:
            set_available(self._keep, True, "")
        self._confidence.set_caption(showing_caption(shown, total))
        self._draw_spread(kept)
        self.shown_changed.emit()

    def unsaved(self) -> tuple[float, float | None] | None:
        """``(shown, saved)`` while the cutoff on screen is not the saved one, else ``None``."""
        if self._behavior_id is None or not self._editable:
            return None
        saved = self._run.state().behavior(self._behavior_id).cutoff
        shown = self.cutoff()
        if saved is not None and abs(float(saved) - shown) < 1e-9:
            return None
        return shown, saved

    def behavior_name(self) -> str:
        if self._behavior_id is None:
            return ""
        return self._run.state().behavior(self._behavior_id).name

    def _back_to_saved(self) -> None:
        """Put the shown cutoff back to the saved one. Writes nothing."""
        saved = self._run.state().behavior(self._behavior_id).cutoff
        if saved is None:
            return
        self._set_cutoff_widgets(float(saved))
        self._preview(float(saved), scroll=True)

    def _draw_spread(self, counts: dict[str, int]) -> None:
        """How many instances each recording keeps at this cutoff (§20).

        It catches a cutoff that keeps most of one animal's behavior and little of another's —
        a bias along the very axis being compared, which the pooled curve cannot show.
        """
        if not counts:
            self._spread.setText("(no recordings searched)")
            return
        peak = max(counts.values())
        typical = sorted(counts.values())[len(counts) // 2]
        out = []
        # By name, which is how the user knows the animal (§11).
        rows = sorted((self._name(rid), n) for rid, n in counts.items())
        width = max(len(name) for name, _n in rows)
        for name, kept in rows:
            bar = "█" * (int(round(12 * kept / peak)) if peak else 0)
            line = f"{name:<{width}} {bar:<12} {kept:>5,}"
            # Named rather than left to the eye: an animal contributing a fraction of the
            # others is the finding this panel is for. Short, to fit the column.
            if typical and kept * 4 <= typical:
                line += " ← far fewer"
            out.append(line)
        self._spread.setText("\n".join(out))

    def _removed_at(self, cutoff: float, behavior) -> int:
        """How many of the clips ``cutoff`` keeps the review removed (§20)."""
        removed = behavior.removed_keys()
        if not removed:
            return 0
        return sum(
            1
            for spec in self._behavior_specs()
            if spec.score >= cutoff
            and (spec.recording_id, int(spec.track_id), spec.start_t_ns, spec.end_t_ns)
            in removed
        )

    # -- the clips -------------------------------------------------------------------- #
    def _behavior_specs(self) -> list:
        """Every candidate of the behavior as a clip, most confident first; built once."""
        specs = self._specs.get(self._behavior_id)
        if specs is None:
            specs = self._specs[self._behavior_id] = self._build_specs()
        return specs

    def _build_specs(self) -> list:
        state = self._run.state()
        removed = frozenset(state.behavior(self._behavior_id).removed_keys())
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
                    candidates,
                    recording_id,
                    subject,
                    state.embed_run_id,
                    timeline,
                    name=self._name(recording_id),
                    removed=removed,
                )
            )
        # Most confident first, and stable between equal scores, so the grid is the ranking.
        specs.sort(key=lambda s: (-float(s.score), s.recording_id, s.start_t_ns))
        return specs

    def _timeline(self, recording_id: str, state):
        """The unit's frame timeline, read once per unit and kept."""
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
            # This unit's embedding is gone; its clips simply do not show. Cached too.
            self._timelines[key] = None
        return self._timelines[key]

    def _read_tiles(self) -> None:
        """Open the preview cache for the searched recordings, off the UI thread, once."""
        run = self._run
        experiment = self._experiment

        def read() -> dict:
            from detail.api import clip_tiles

            state = run.state()
            rids = list(state.search_recording_ids)
            embeds = {rid: state.embed_run_id for rid in rids}
            tracking = clip_tiles.tracking_runs(  # off-thread (run_busy)
                experiment.dir, embeds
            )
            return {
                "embeds": embeds,
                "tracking": tracking,
                "missing": clip_tiles.missing(  # off-thread (run_busy)
                    experiment, tracking, rids
                )["proxy"],
            }

        run_busy(
            self,
            read,
            self._on_tiles_read,
            label="Opening the clips' previews…",
            supersedes="calibrate_tiles",
        )

    def _on_tiles_read(self, data: dict) -> None:
        from detail.api import clip_tiles

        self._catalog = clip_tiles.TileCatalog(
            self._experiment, data["tracking"], data["embeds"]
        )
        self._missing = list(data["missing"])
        self._tiles_read = True
        if self._behavior_id is not None:
            self._show_tiles(new_source=True, scroll=True)

    def _show_tiles(self, *, new_source: bool = False, scroll: bool = False) -> None:
        """Hand the grid every clip, split at the shown cutoff by a labelled dashed rule.

        Most confident first, the dropped clips from the row after the rule. A stepped line
        through one unbroken grid moved no tile but was too easy to miss; the rule is straight,
        in its own band, and says in words which side is which. Nothing below it is faded: those
        clips are still to be watched. A clip AUDIT removed is faded and crossed as it is there,
        and a clip on the behavior's own example or non-example says so.
        """
        from detail.api import clip_tiles

        if self._behavior_id is None or not self._tiles_read:
            return
        specs = self._behavior_specs()
        cutoff = self.cutoff()
        behavior = self._run.state().behavior(self._behavior_id)
        removed = frozenset(behavior.removed_keys())
        told = query_runs.told_kinds([behavior])
        context = self._montage.context_s()
        # Enough digits to tell neighbours apart, but never more than three: the caption is
        # 150px, and the recording's name after the score is what a long score cut off.
        digits = min(score_digits([s.score for s in specs]), 3) if specs else 2
        tiles = []
        n_kept = 0
        for rank, spec in enumerate(specs, start=1):
            n_kept += float(spec.score) >= cutoff
            gone = (
                spec.recording_id,
                int(spec.track_id),
                spec.start_t_ns,
                spec.end_t_ns,
            ) in removed
            # What the user told the method, found again: the search scores its own input at
            # the top, and a reader judging the cutoff should know those are not finds.
            kind = query_runs.told_kind(
                told,
                behavior.id,
                spec.recording_id,
                spec.track_id,
                spec.start_t_ns,
                spec.end_t_ns,
            )
            low, high = context_span(spec.start_t_ns, spec.end_t_ns, context)
            tiles.append(
                MontageClip(
                    max(high - low, _MIN_LOOP_NS),
                    rank,
                    # Score, time, then name: a caption cut short loses the end of the name,
                    # which the tooltip still has, rather than the time.
                    f"{spec.score:.{digits}f} · {_clock(spec.start_t_ns)} · "
                    f"{self._name(spec.recording_id)}",
                    tag="removed" if gone else (f"your {kind}" if kind else None),
                    faded=gone,
                    note=(
                        "Removed in Audit: not in this behavior's instances."
                        if gone
                        else (
                            f"This is your {kind} clip, which the search was given — it "
                            "finds it, so it is not a find."
                            if kind
                            else ""
                        )
                    ),
                    crossed=gone,
                )
            )
        here = _cutoff_text(cutoff)
        dropped = len(specs) - n_kept
        saved = behavior.cutoff
        unsaved = saved is None or abs(float(saved) - cutoff) >= 1e-9
        rule = (
            f"▲ {n_kept:,} kept   ·   cutoff {here}{' (not saved)' if unsaved else ''}   ·   "
            f"{dropped:,} dropped ▼"
        )
        if n_kept and dropped:
            sections = [("", n_kept), (rule, dropped)]
        elif dropped:
            sections = [(rule, dropped)]  # nothing kept: the rule heads the whole grid
        else:
            sections = None  # everything kept: there is no line to draw
        same = self._source_for == self._behavior_id and not new_source
        self._montage.set_clips(tiles, keep_scroll=same, sections=sections)
        self._kept_count = n_kept
        if not same and self._catalog is not None:
            self._montage.set_source(
                clip_tiles.ClipTiles(
                    self._catalog,
                    [
                        clip_tiles.TileClip(
                            spec.recording_id,
                            int(spec.track_id),
                            *context_span(spec.start_t_ns, spec.end_t_ns, context),
                        )
                        for spec in specs
                    ],
                )
            )
            self._source_for = self._behavior_id
        self._sync_grid_note(specs)
        if scroll:
            self._scroll_to_cutoff(n_kept)
        self._sync_below_bar()

    def _sync_grid_note(self, specs: list) -> None:
        missing = [self._name(rid) for rid in self._missing]
        if not specs:
            text = "This behavior's search found nothing to show."
        elif missing:
            text = (
                f"No preview yet for {', '.join(missing)}: their clips cannot play until the "
                "previews are built (detail cache previews build)."
            )
        else:
            text = ""
        self._grid_note.setText(text)
        self._grid_note.setVisible(bool(text))

    def _sync_below_bar(self, *_args) -> None:
        """Pin a bar to the top of the grid while every clip in view is below the cutoff.

        The line is easy to scroll past in a long ranking, and below it the clips look like any
        other; the bar says where you are and takes you back. With nothing kept, the whole grid is
        below the cutoff and the bar says so, with no line to go back to.
        """
        bar = self._below_bar
        specs = self._behavior_specs() if self._behavior_id is not None else []
        kept = self._kept_count
        dropped = len(specs) - kept
        viewport = self._montage.grid.viewport()
        here = _cutoff_text(self.cutoff())
        show = False
        if specs and kept == 0:
            show = True
            bar.set_text(
                f"Nothing is kept at {here}: every clip here is below the cutoff",
                back=False,
            )
        elif dropped > 0:
            headings = self._montage.heading_rects()
            if headings:
                lowest = headings[-1][1].bottom()
                top = self._montage.grid.verticalScrollBar().value()
                show = top > lowest
                bar.set_text(
                    f"Below the cutoff ({here}): these {dropped:,} clips are dropped",
                    back=True,
                )
        bar.setVisible(show)
        if show:
            bar.setGeometry(0, 0, viewport.width(), bar.sizeHint().height())
            bar.raise_()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 — Qt's name
        if (
            watched is self._montage.grid.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            self._sync_below_bar()
        return super().eventFilter(watched, event)

    def _scroll_to_cutoff(self, n_kept: int) -> None:
        """Bring the cutoff into view: the last kept row a third of the way down the grid."""
        if not self._montage.clips():
            return
        rect = self._montage.tile_rect(max(n_kept - 1, 0))
        bar = self._montage.grid.verticalScrollBar()
        bar.setValue(max(rect.top() - self._montage.grid.viewport().height() // 3, 0))

    def _hover_text(self, index: int) -> str:
        specs = self._behavior_specs() if self._behavior_id is not None else []
        if not 0 <= index < len(specs):
            return ""
        spec = specs[index]
        return (
            f"{self._name(spec.recording_id)} · {format_time(spec.start_t_ns)} · "
            f"confidence {float(spec.score):.5f}"
        )

    def _click_hint(self, index: int) -> str:
        specs = self._behavior_specs() if self._behavior_id is not None else []
        if not 0 <= index < len(specs):
            return ""
        return f"Click to show the cutoff at this clip ({_cutoff_text(specs[index].score)})"

    def _on_tile_clicked(self, index: int, _shift: bool) -> None:
        """A clip clicked: show the cutoff at it, so it is the last one kept. Not saved."""
        specs = self._behavior_specs()
        if not self._editable or not 0 <= index < len(specs):
            return
        value = float(specs[index].score)
        # The rows re-flow when the rule moves; keep the clicked clip at the height it was
        # clicked at, so what is under the pointer is still the clip just chosen.
        scroll = self._montage.grid.verticalScrollBar()
        at = self._montage.tile_rect(index).top() - scroll.value()
        self._set_cutoff_widgets(value)
        self._preview(value, scroll=False)
        scroll.setValue(max(self._montage.tile_rect(index).top() - at, 0))

    def _play_clip(self, index: int) -> None:
        """A tile double-clicked (or Space): the clip in the clip window, in grid order."""
        from detail.gui import clip_popup

        specs = self._behavior_specs() if self._behavior_id is not None else []
        if not 0 <= index < len(specs):
            return
        clip_popup.open_clip_window(
            self,
            self._experiment,
            specs[index],
            sequence=specs,
            index=index,
            context_s=1,
            order_note="Most confident first",
        )

    def export_clips(self) -> list:
        """The clips the shown cutoff keeps, in grid order, for **Export ▾**."""
        from detail.api import clip_montage as montage_api

        if self._behavior_id is None:
            return []
        cutoff = self.cutoff()
        return [
            montage_api.MontageClip(
                rank=rank,
                recording_id=spec.recording_id,
                recording_name=self._name(spec.recording_id),
                subject=int(spec.track_id),
                embed_run_id=spec.embed_run_id,
                start_t_ns=spec.start_t_ns,
                end_t_ns=spec.end_t_ns,
            )
            for rank, spec in enumerate(self._behavior_specs(), start=1)
            if float(spec.score) >= cutoff
        ]

    def _export_stem(self) -> str:
        if self._behavior_id is None:
            return "behavior"
        name = self._run.state().behavior(self._behavior_id).name
        return f"{name}_from_{_cutoff_text(self.cutoff())}"

    # -- the cutoff, from any control ------------------------------------------------- #
    def _set_cutoff_widgets(self, value: float) -> None:
        self._syncing = True
        try:
            self._line.setValue(float(value))
            self._confidence.set_value(float(value))
        finally:
            self._syncing = False

    def _preview(self, value: float, *, scroll: bool) -> None:
        """Show what ``value`` would keep. Writes nothing."""
        if self._behavior_id is None:
            return
        self._refresh_header(value)
        self._show_tiles(scroll=scroll)

    def _on_line_moved(self) -> None:
        if self._syncing:
            return
        value = self.cutoff()
        self._syncing = True
        try:
            self._confidence.set_value(value)
        finally:
            self._syncing = False
        self._preview(value, scroll=True)

    def _on_confidence_typed(self, value: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._line.setValue(float(value))
        finally:
            self._syncing = False
        self._preview(float(value), scroll=True)

    def _on_keep(self) -> None:
        self._commit(self.cutoff())

    def _commit(self, value: float) -> None:
        """Save the cutoff and reproject — off the UI thread, and only from **Keep**."""
        if not self._editable or self._behavior_id is None:
            return
        behavior_id = self._behavior_id
        run = self._run

        def work(emit):
            run.set_cutoff(behavior_id, float(value), progress=emit)

        run_progress(
            self,
            work,
            lambda _r: self._on_committed(value),
            label="Saving the cutoff…",
        )

    def _on_committed(self, value: float) -> None:
        self._refresh_header(value)
        # The rule's "(not saved)" is part of the grid, so the grid hears it too.
        self._show_tiles()
        # The other tabs read the artifact this just rewrote.
        self.committed.emit()


class _MeasuresView(QWidget):
    """MEASURES on a query run: the §23 table, then where a behavior's instances land and how
    they compare with the recording.

    The map and the comparison used to be bands under CALIBRATE, reading the saved cutoff's
    artifact under a page about choosing one. They answer "what did this behavior turn out to
    be", which is this view's question, so they are here; CALIBRATE is the curve and the clips.
    The §23 table reads the live artifact, so a snapshot shows the two bands without it.
    """

    def __init__(self, run, experiment) -> None:
        super().__init__()
        self._run = run
        self._experiment = experiment
        self._behavior_id: str | None = None
        self._outlines_by_recording: dict = {}
        self._frame_aspects: dict = {}
        self._embed_run_id = ""
        #: What the map last drew: ``{recording_id: [placement, ...]}``; ``None`` for a run that
        #: never recorded positions.
        self._placements_by_recording: dict | None = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if run.editable:
            from detail.gui.pages import (
                MeasuresPanel,
            )  # local: pages imports this module

            layout.addWidget(MeasuresPanel(experiment, run.id))
        # Where those instances land — a cutoff that keeps a behavior only in one corner is a
        # spatial bias no count shows.
        self._map = SpatialMap(experiment)
        self._map_band = CollapsibleBand(
            "Locations", self._map, key="query/locations", on_resize=self._map.resize_by
        )
        layout.addWidget(self._map_band)
        # Against the **recording**, not the other behaviors: a query's matches are a selection
        # from the footage rather than a partition of it, and most query runs have one behavior.
        self._measurements = MeasurementsPanel(
            experiment,
            "matches",
            "behavior",
            key="query/measurements",
            baseline=BASELINE_RECORDING,
        )
        self._measurement_band = CollapsibleBand(
            "Interpretation",
            self._measurements,
            key="query/measurements",
            on_resize=self._measurements.resize_by,
        )
        layout.addWidget(self._measurement_band)
        layout.addStretch(1)

    def set_behavior(self, behavior_id: str) -> None:
        self._behavior_id = behavior_id
        self._measurements.select(self._run, behavior_id)
        self.refresh()

    def recheck(self) -> None:
        """A cutoff was saved or a clip removed: the matches changed under both bands."""
        self._measurements.recheck()
        self.refresh()

    def refresh(self) -> None:
        run_busy(
            self,
            self._placements,
            self._show_placements,
            label="Reading instances…",
            supersedes="query_measures",
        )

    def _placements(self) -> dict | None:
        """Off-thread: where each kept instance happened, with each recording's field of view.

        ``None`` means the run never recorded positions (fixable); ``{}`` means it did and none
        landed. Carried apart so the map can say which.
        """
        from detail.storage import behavior_output as bo

        try:
            artifact = self._run.artifact()  # off-thread (run_busy)
        except FileNotFoundError:
            return {}
        instances = [
            row
            for row in bo.bouts(artifact.instances)
            if self._behavior_id is None or row.source_label == self._behavior_id
        ]
        outlines: dict = {}
        aspects: dict = {}
        for recording_id in sorted({row.recording_id for row in instances}):
            try:
                recording = self._experiment.get_recording(recording_id)
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
        self._embed_run_id = self._run.state().embed_run_id
        features = artifact.features
        if features is None:
            return None
        # Placements, not bare coordinates: a dot that reports only where it was cannot be
        # turned back into the clip it stands for, which is the whole of clicking one.
        placements: dict = {}
        for row in instances:
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
        return placements

    def _show_placements(self, placements) -> None:
        self._placements_by_recording = placements
        if placements is None:
            self._map_band.setVisible(True)
            self._map.show_unavailable(
                "This run has no recorded positions — it was made before position capture. "
                "Run `detail recording backfill-kinematics` and re-open the run to fill it in."
            )
            return
        # An absence is not the claim "it happens nowhere": no box when nothing landed.
        self._map_band.setVisible(bool(placements))
        if placements:
            self._map.show_instances(
                placements,
                outlines=self._outlines_by_recording,
                aspects=self._frame_aspects,
                embed_run_ids={rid: self._embed_run_id for rid in placements},
                colour=chart_theme.rgb(plot_theme.current().identity(0)),
            )


QUERY_VIEWS = (
    (
        "Calibrate",
        "Set each behavior's cutoff: watch the clips it keeps and drops, then keep it.",
    ),
    ("Audit", "Review every clip the cutoff keeps, and remove the wrong ones."),
    ("Browse", "Every find above the cutoff, playing one at a time."),
    (
        "Measures",
        "Time, bouts and transitions per behavior, where they happen and how they move.",
    ),
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
        # One line: which behavior, and the run's state. The picker used to be a tab bar the
        # width of the window holding one tab per behavior — a 40px bar reading "rear" across
        # 2,000px on the common one-behavior run, and a second row of tabs under the views'.
        top = QHBoxLayout()
        top.addWidget(QLabel("Behavior"))
        self._selector = QComboBox()
        self._selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._behaviors = list(self._run.behaviors())
        for behavior in self._behaviors:
            self._selector.addItem(behavior.name, behavior.id)
        self._selector.currentIndexChanged.connect(self._on_behavior_changed)
        top.addWidget(self._selector)
        top.addSpacing(lines(1))
        self._status = QLabel("Loading search results…")
        self._status.setWordWrap(True)
        top.addWidget(self._status, 1)
        root.addLayout(top)

        # Over every view but CALIBRATE, while CALIBRATE holds a cutoff that is not saved: where
        # the previewed cutoff was being mistaken for the applied one, with the one-click fix.
        self._unsaved_notice = _AttentionBar()
        self._notice_keep = QPushButton("Keep")
        self._notice_keep.setObjectName("unsaved_notice_keep")
        self._notice_to_calibrate = QPushButton("Go to Calibrate")
        self._notice_to_calibrate.setObjectName("unsaved_notice_to_calibrate")
        self._unsaved_notice.add_button(self._notice_keep)
        self._unsaved_notice.add_button(self._notice_to_calibrate)
        self._unsaved_notice.hide()
        root.addWidget(self._unsaved_notice)

        self._tabs = QTabWidget()
        self._calibrate = _CalibrateTab(self._run, experiment)
        self._audit = QueryReviewTab(experiment, self._run)
        # Clustering's BROWSE, not a copy of it: every behavior with a saved cutoff at once,
        # the ethogram across the top, the list beside a player (§20).
        self._browse = LabelBrowseTab(self._run, experiment, QUERY_TERMS)
        # Where the matches land and how they move: MEASURES, not CALIBRATE (`_MeasuresView`).
        self._measures = _MeasuresView(self._run, experiment)
        # The tab is a leaf and does not know this workspace, so it asks rather than reaching
        # for the selector; driving the selector keeps the two in sync in both directions.
        # A removal or a finish rewrites the artifact Browse reads, and Calibrate's marks.
        self._audit.changed.connect(self._browse.reload)
        self._audit.changed.connect(self._calibrate.forget_clips)
        self._audit.changed.connect(self._measures.recheck)
        # A saved cutoff rewrites the artifact the other tabs read.
        self._calibrate.committed.connect(self._browse.reload)
        self._calibrate.committed.connect(self._audit.reload)
        self._calibrate.committed.connect(self._measures.recheck)
        # A saved cutoff can be the last one Finish was waiting for.
        self._calibrate.committed.connect(self._sync_status)
        self._calibrate.shown_changed.connect(self._sync_unsaved_notice)
        self._tabs.currentChanged.connect(lambda _i: self._sync_unsaved_notice())
        self._notice_keep.clicked.connect(self._calibrate._on_keep)
        self._notice_to_calibrate.clicked.connect(lambda: self.show_view("Calibrate"))
        # AUDIT with nothing to review offers the one step that would give it something.
        self._audit.go_to_calibrate.connect(lambda: self.show_view("Calibrate"))
        self._audit.reopen_requested.connect(self._on_reopen)
        self._tabs.addTab(self._calibrate, "Calibrate")
        self._tabs.addTab(self._audit, "Audit")
        self._tabs.addTab(self._browse, "Browse")
        # The §23 numbers as a tab, not a section under every tab (see `ClusterWorkspace`).
        self._tabs.addTab(self._measures, "Measures")
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
        # Why Finish is greyed, in words beside it rather than only in a tooltip: a run finished
        # with a behavior uncut was the dead end AUDIT then described twice at once.
        self._finish_note = QLabel("")
        self._finish_note.setWordWrap(True)
        set_label_color(self._finish_note, MUTED)
        actions.addWidget(self._finish_note, 1)
        root.addLayout(actions)

        self._sync_status()
        self._sync_unsaved_notice()
        if self._behaviors:
            self._load_behavior(self._behaviors[0].id)

    def _sync_unsaved_notice(self) -> None:
        """Show the notice over a view other than CALIBRATE while CALIBRATE's cutoff is not saved."""
        unsaved = self._calibrate.unsaved()
        show = unsaved is not None and self._tabs.currentWidget() is not self._calibrate
        if show:
            shown, saved = unsaved
            name = self._calibrate.behavior_name()
            self._unsaved_notice.text.setText(unsaved_notice_text(name, shown, saved))
            self._notice_keep.setText(f"Keep {_cutoff_text(shown)}")
        for button in (self._notice_keep, self._notice_to_calibrate):
            set_available(button, show, "Nothing on Calibrate is waiting to be saved.")
        self._unsaved_notice.setVisible(show)

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
            names = experiment.recording_names() if experiment is not None else {}
            return run.frequency_curve(behavior_id, steps=_CURVE_STEPS), names

        experiment = self._experiment
        run_progress(
            self,
            work,
            lambda result: self._on_loaded(behavior_id, *result),
            label="Loading search results…",
            # Switching behavior mid-load: the older one landing last showed it, not the choice.
            supersedes="results",
        )

    def _on_loaded(self, behavior_id: str, curve, names: dict[str, str]) -> None:
        self._curves[behavior_id] = list(curve)
        self._calibrate.set_names(names)
        self._show_behavior(behavior_id)

    def _show_behavior(self, behavior_id: str) -> None:
        self._calibrate.set_behavior(behavior_id, self._curves.get(behavior_id, []))
        self._audit.set_behavior(behavior_id)
        self._measures.set_behavior(behavior_id)
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
        # The API's own refusal, read before the press (`QueryRun.finish_blocker`): one rule and
        # one wording for the button, `finalize` and the CLI.
        blocker = self._run.finish_blocker() if editable and not finished else None
        set_available(
            self._finish,
            editable and not finished and blocker is None,
            frozen if not editable else (blocker or ""),
        )
        set_available(
            self._reopen,
            editable and finished,
            (
                "The run is open; Reopen applies once it is finished."
                if editable
                else frozen
            ),
        )
        self._finish_note.setText(blocker or "")
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
