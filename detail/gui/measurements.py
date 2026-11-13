"""What a cluster or a behavior *is*, as numbers — one shared band for both result pages.

The question this answers: given eight behaviors, how do you name them? Naming clusters by eye
from their clips is no better than eyeballing a UMAP of them, and the information component of
that judgement had not been automated.

The first version drew every measurement, equally, in a row of its own. That is a scanning
layout with nothing to scan for: on a real run, one or two measurements carry most of
the between-cluster variance, and most clusters are distinctive on few of them or none.
Equal plots bury the few that matter under the many that do not. With that many plots,
a user needs to pick out
quickly which measurements matter and then dig into those.

So the band is a **scan and a drill-in**, not a wall:

1. One line about the *run* — which measurements this clustering separates at all. It does not
   move when the selection does, because it is a property of the partition.
2. A table of **every** measurement for the selection, ranked, with its value, where it sits, and
   the effect size. A table rather than prose: asked for objective, and "ordinary on everything
   else" is half the description, which only the complete list carries.
3. The **top few drawn**, at a readable height with a labelled axis.
4. Click one to open it alone, with the other clusters' medians in their own lane above it.

One widget for both surfaces, like ``SpatialMap`` beside it: a measurement that answers a
clustering run answers a behavior scan, and a band on one page and not the other is the class
``tests/gui/test_workspace_parity.py`` exists to catch.

**The player arrives with something to play.** It used to hold a third of the width empty from
the moment the tab opened, which is why the plots were thin — the same failure ``gui_design``
already names: *a surface with nothing in it yet shows the one action, not the empty frame*.

**Height is the user's, not ours.** A row's height trades how much of a spread you can see
against how many fit on screen, and which matters depends on whether you are scanning or
studying. So the band drags, and remembers.

**Measured when asked, and never drawn from a measurement the run has moved past** (#515).
Measuring reads every unit's track — over a minute on a large run — and it used to start
on the first click, mid-interpretation, then be kept for the life of the panel under the cluster
ids it was taken at. A granularity change or a name moves those ids, so a cluster new to the cut
partitioned the old rows into nothing: the run-wide spread still drew, the members were quietly
missing. Now the panel measures on a button, keeps what the rows were read *from*, and checks it
against the run on every draw; when they differ it says so and offers to measure again. The
check is by what the artifact states, not by which edit ran: "a granularity change only
relabels" holds for ``dtw_finch`` alone, while ``codebook_viterbi`` (and the removed pose method
``finch``) re-encode their runs at every cut, so most spans are new ones.
"""

import time
from typing import NamedTuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from detail.analysis.descriptor_ranking import (
    STRONG,
    rank_run,
    rank_selection,
    separation_withheld,
)
from detail.analysis.descriptors import DESCRIPTORS
from detail.core import run_selection
from detail.gui.behavior_query import ClipSpec, _ClipPlayer
from detail.gui.busy import run_busy, run_progress
from detail.gui.grip_splitter import GripSplitter
from detail.gui.lifetime import discard
from detail.gui.plots import (
    DescriptorStrip,
    MAX_ROW_PX,
    MIN_ROW_PX,
    cluster_lane,
    fmt_value,
)
from detail.gui.sizing import lines, remember_size, remembered_size
from detail.gui.tables import give_slack_to, row_id

#: How many measurements are drawn without asking. Three is what fits above the fold beside the
#: table; the rest are one click away, and the table already names them all.
DRAWN = 3

#: What "the rest of the run" is measured over. No default anywhere: it decides whether the
#: comparison is right, and a default spelling "as before" is where a missing fact hides — the
#: rule the shared-widget guard enforces.
BASELINE_INSTANCES = "instances"
BASELINE_RECORDING = "recording"
BASELINES = frozenset({BASELINE_INSTANCES, BASELINE_RECORDING})

#: How many background spans to measure per recording and subject. The pool only has to describe
#: a distribution, and a few hundred fill a 48-bin histogram.
BASELINE_WINDOWS = 400

_NS_PER_S = 1_000_000_000

#: Ordinary buttons, saying what they do. Constants and free of any count, because the snapshot
#: guard keys its allowlist on a button's text and a label that moves with the data cannot be
#: listed.
DRAW_ALL = "Draw all measurements"
SHOW_FEWER = "Draw only the distinctive ones"
BACK_TO_ALL = "All measurements"
MEASURE = "Measure this run"
REMEASURE = "Re-measure"

#: The ranked table's columns. ``_table(navigable=True)`` appends the trailing "›" itself.
TABLE_HEADERS = ["measurement", "value", "vs the rest", "effect", "", "drawn"]

#: How many are drawn when *none* of them is strong. Fewer, because the picture is then about
#: the absence rather than about any one measurement, and the note above already said so.
WEAK_DRAWN = 2

#: A row's height in lines of the interface font, before any drag. Enough for the dots to stack
#: and for the axis beneath them — one line is a rule with marks on it, which is what shipped.
DEFAULT_ROW_LINES = 6


def _finite(values) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    return values[np.isfinite(values)]


def _label(
    text: str, *, bold: bool = False, muted: bool = False, wrap: bool = True
) -> QLabel:
    widget = QLabel(text)
    widget.setWordWrap(wrap)
    if bold:
        font = QFont(widget.font())
        font.setBold(True)
        widget.setFont(font)
    if muted:
        from detail.gui.pages import MUTED, set_label_color

        set_label_color(widget, MUTED)
    return widget


def _name_column_px() -> int:
    """Wide enough for the longest measurement name in the interface font, plus air."""
    from PyQt6.QtGui import QFontMetrics

    metrics = QFontMetrics(QLabel().font())
    longest = max(
        (metrics.horizontalAdvance(d.label) for d in DESCRIPTORS.values()), default=80
    )
    return int(longest + 24)


class MeasurementsPanel(QWidget):
    """The band: a ranked scan of every measurement, with the top few drawn.

    **Against the run, never against itself.** A first draft drew each selection's spread from
    its own values, which renders everything as a full-width distribution centred on itself — a
    picture that looks informative and says nothing. What makes "1.8 px/frame" mean *slow* is the
    other seven hundred bouts to the right of it. The degenerate form of that is a run with a
    single group with nothing else measured, where the two sets are literally equal — which is
    why a run whose instances are a *selection* from its recordings (a query's matches) measures
    background spans too, and says so.

    **The rows come from the registry**, never a list here: a panel naming its own descriptors
    would show `duration` on a pose cluster, where a member is one frame. The module declares
    what each descriptor needs and which levels it means anything at.

    Measuring is O(instances) and reads zarr, so it happens **when asked**, with a bar, and is
    filtered per selection afterwards. Every draw first checks the run still states what was
    measured (:class:`Measured`); an edit that changed it leaves the band saying so, with the
    button to measure again, rather than drawing the old rows under the new ids.
    """

    instance_picked = pyqtSignal(object)

    def __init__(
        self, experiment, member_noun: str, unit_noun: str, *, key: str, baseline: str
    ) -> None:
        super().__init__()
        if baseline not in BASELINES:
            raise ValueError(
                f"unknown baseline {baseline!r}; expected one of {sorted(BASELINES)}"
            )
        self._baseline = baseline
        self._experiment = experiment
        self._member_noun = member_noun
        self._unit_noun = unit_noun
        self._key = key
        self._row_px = remembered_size(key, lines(DEFAULT_ROW_LINES))
        self._measured: Measured | None = None
        self._run_id: str | None = None
        self._run = None
        self._measuring = False
        # Bumped by every selection, so a check that lands after a later one is dropped rather
        # than drawing the cluster the user has already left.
        self._generation = 0
        self._rows: list = []
        self._members: dict = {}
        self._values: dict = {}
        self._labels = np.asarray([], dtype=object)
        self._selected = None
        #: The selected group as a person names it (a behavior's name, not its id).
        self._selected_name: str | None = None
        self._ranked: list = []
        self._drawn_all = False
        self._open: str | None = (
            None  # the measurement drilled into, or None for the scan
        )
        self._player: _ClipPlayer | None = None
        #: recording id -> the embedding run a clicked member plays through (§6.5).
        self._embed_run_ids: dict[str, str] = {}
        self.posture_source: str | None = None

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)

        self._run_line = _label("", muted=True)
        box.addWidget(self._run_line)
        self._heading = _label("", bold=True)
        box.addWidget(self._heading)
        self._note = _label(f"Select a {unit_noun} to see what is in it.")
        box.addWidget(self._note)
        # Built hidden and labelled, for the same reason as the two below.
        self._measure_button = QPushButton(MEASURE)
        self._measure_button.hide()
        self._measure_button.clicked.connect(self.measure_now)
        measure_row = QHBoxLayout()
        measure_row.setContentsMargins(0, 0, 0, 0)
        measure_row.addWidget(self._measure_button)
        measure_row.addStretch(1)
        box.addLayout(measure_row)
        # A real table, carrying the package's own cue that a row opens something: pointing hand,
        # trailing "›", double-click (`gui_design` §3). Every row becomes an entry point, where
        # before there was one control that read as text — reported as hard to find because the
        # render-more button looked like none of the other buttons, and `setFlat`
        # appears nowhere else in `detail/gui/`, so it looked like nothing else because it was.
        self._table_box = QVBoxLayout()
        self._table_box.setContentsMargins(0, 0, 0, 0)
        self._table = None
        box.addLayout(self._table_box)

        self._caption = _label("", muted=True, wrap=False)
        # Built hidden and labelled. A control with no text yet is the hole an allowlist keyed
        # on labels cannot see — it matches the empty string — and neither of these means
        # anything before there is something drawn.
        self._more = QPushButton(DRAW_ALL)
        self._more.hide()
        self._more.clicked.connect(self._toggle_all)
        self._back = QPushButton(BACK_TO_ALL)
        self._back.hide()
        self._back.clicked.connect(lambda: self._open_measurement(None))
        caption_row = QHBoxLayout()
        caption_row.setContentsMargins(0, 0, 0, 0)
        caption_row.addWidget(self._back)
        caption_row.addWidget(self._more)
        caption_row.addWidget(self._caption)
        caption_row.addStretch(1)
        box.addLayout(caption_row)

        # Rows on the left; the player joins them on the right the first time something is
        # clicked. Built empty and grown, so nothing holds width it has no use for.
        self._plots = QWidget()
        self._plots_box = QVBoxLayout(self._plots)
        self._plots_box.setContentsMargins(0, 0, 0, 0)
        self._plots.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        # Neither side folds: the rows are the page, and the player is what they open.
        self._split = GripSplitter(Qt.Orientation.Horizontal, key="measurements")
        self._split.addWidget(self._plots)
        box.addWidget(self._split, 1)

        self._legend = _label("", muted=True)
        box.addWidget(self._legend)
        self._played = _label("", muted=True)
        self._played.hide()
        box.addWidget(self._played)

    # -- the band's drag ----------------------------------------------------------------- #
    def resize_by(self, delta: int) -> None:
        """Drag down for taller rows, up for shorter — the band's handle calls this.

        Taller trades how many measurements fit on screen against how much of each spread you
        can see, and which of those matters depends on whether you are scanning the rows or
        studying one of them.
        """
        self._row_px = int(min(MAX_ROW_PX, max(MIN_ROW_PX, self._row_px + delta)))
        remember_size(self._key, self._row_px)
        for _name, row in self._rows:
            row.set_row_height(self._row_px)

    # -- rendering ------------------------------------------------------------------------ #
    def show_measurements(
        self, run_wide: dict, members: dict, member_ids: dict, labels=None
    ) -> None:
        """Draw the band.

        ``run_wide[name]`` is every instance's value, ``members[name]`` the selection's, and
        ``member_ids[name][i]`` identifies the member behind dot ``i``. ``labels`` is the group
        each run-wide value belongs to, which is what makes the two rankings possible; without it
        the band still draws, unranked, in registry order.
        """
        self._values = run_wide
        self._members = member_ids
        self._member_values = members
        self._labels = np.asarray(labels if labels is not None else [], dtype=object)
        self._open = None
        self._drawn_all = False
        self._rank()
        self._render()

    def _rank(self) -> None:
        """The two rankings, which fail on different conditions and are withheld separately.

        The run line needs a **partition** — two groups for the grouping to explain anything. The
        selection needs a non-empty **complement**, which background rows supply even when the run
        has one group. Tying them together is what left four of five real query runs with an empty
        band.
        """
        self._ranked, self._run_ranked, self._withheld = [], [], None
        if not self._values or not len(self._labels):
            return
        self._withheld = separation_withheld(
            [self._group_name(label) for label in self._labels], self._unit_noun
        )
        if self._withheld is None:
            self._run_ranked = rank_run(self._values, self._labels)
        if self._selected is not None:
            self._ranked = rank_selection(self._values, self._labels, self._selected)

    def _render(self) -> None:
        self._clear_rows()
        self._measure_button.hide()
        if not self._values:
            self._run_line.setText("")
            self._heading.setText("")
            self._show_table([])
            self._caption.setText("")
            self._more.hide()
            self._back.hide()
            self._note.setText(
                f"No measurements for this {self._unit_noun}. Its run was made before position "
                "capture — run `detail recording backfill-kinematics` and re-open it."
            )
            self._note.show()
            return
        self._run_line.setText(self._run_sentence())
        self._heading.setText(self._heading_text())
        self._back.setVisible(self._open is not None)
        if self._open is not None:
            self._draw_one(self._open)
        else:
            self._draw_scan()

    def _run_sentence(self) -> str:
        """Which measurements this run's groups differ on at all — the same for every selection."""
        if self._withheld is not None:
            return self._withheld
        if not self._run_ranked:
            return ""
        top = ", ".join(
            f"{DESCRIPTORS[name].label} {value:.2f}"
            for name, value in self._run_ranked[:3]
        )
        tail = ", ".join(DESCRIPTORS[name].label for name, _v in self._run_ranked[-2:])
        groups = len(set(self._labels.tolist()))
        return (
            f"Across this run's {groups} {self._unit_noun}s, the measurements that separate them "
            f"most are {top} (share of variance explained); least, {tail}."
        )

    def _heading_text(self) -> str:
        count = len(next(iter(self._member_values.values()), []))
        name = "" if self._selected is None else f" {self._selected_name}"
        if self._open is not None:
            return (
                f"{DESCRIPTORS[self._open].label} · {self._unit_noun}{name} · "
                f"{count} {self._member_noun}"
            )
        return f"{self._unit_noun.capitalize()}{name} · {count} {self._member_noun}"

    def _draw_scan(self) -> None:
        strong = [r for r in self._ranked if r.strong]
        chosen = self._ranked if self._drawn_all else strong[:DRAWN]
        if not self._ranked:
            self._note.setText(
                self._withheld or f"Select a {self._unit_noun} to see what is in it."
            )
            self._note.show()
        elif not strong:
            # Both halves follow the baseline: "the rest of the run" is not what a query is
            # compared against, and "split it at a finer granularity" is a control a query does
            # not have. A remedy naming a knob the page lacks is worse than no remedy.
            against = (
                "the rest of the recording"
                if self._baseline == BASELINE_RECORDING
                else "the rest of the run"
            )
            remedy = (
                "Name it from the clips, or narrow the search."
                if self._baseline == BASELINE_RECORDING
                else "Name it from the clips, or split it at a finer granularity."
            )
            self._note.setText(
                f"No measurement separates this {self._unit_noun} from {against} — the largest "
                f"effect is {DESCRIPTORS[self._ranked[0].name].label} at delta "
                f"{self._ranked[0].delta:+.2f}, under the {STRONG} threshold. {remedy}"
            )
            self._note.show()
            chosen = self._ranked if self._drawn_all else self._ranked[:WEAK_DRAWN]
        else:
            self._note.hide()
        # The table is marked from the *same* list the plots are built from. Deriving "what is
        # drawn" a second time is how it came to tick three rows under a caption reading "the 2
        # largest effects", with two plots on screen — every number right, the marks wrong.
        self._show_table(chosen)
        for ranked in chosen:
            self._add_row(ranked.name, ranked)
        self._caption.setText(self._scan_caption(len(chosen), len(strong)))
        self._more.setVisible(len(self._ranked) > len(chosen) or self._drawn_all)
        self._more.setText(SHOW_FEWER if self._drawn_all else DRAW_ALL)
        self._finish_plots()

    def _draw_one(self, name: str) -> None:
        """One measurement, with the other groups' medians in a lane of their own above it.

        Two pictures rather than one: a tick in the lane is a whole {unit}, a mark in the plot
        is one {member}, and putting both on one axis is what made the first version unreadable.
        """
        self._show_table([])
        self._note.hide()
        # The plot first, then the lane **from the plot's own range**. Both were computing
        # "the same" robust range independently and getting different answers — the strip pools
        # the members in with the run, and this cluster's are the extreme ones, so its 99th
        # percentile sat at 3.009 against the lane's 2.961. Identical plotting areas, different
        # scales: the two drifted apart by 15px across the axis, so a tick stopped sitting over
        # its own data while both plots rendered perfectly.
        strip = self._add_row(name, None, height=self._row_px * 3, bare=True)
        low, high = strip.x_range
        member_noun = (
            self._member_noun[:-1]
            if self._member_noun.endswith("s")
            else self._member_noun
        )
        for index, widget in enumerate(
            (
                _label(
                    f"every {self._unit_noun} in the run — one tick each, at its median",
                    muted=True,
                ),
                cluster_lane(
                    self._group_medians(name), self._own_median(name), low, high
                ),
                _label(
                    f"every {member_noun} in the run — this {self._unit_noun}'s in colour",
                    muted=True,
                ),
            )
        ):
            self._plots_box.insertWidget(index, widget)
        self._caption.setText(
            f"Two pictures, two units: the strip counts {self._unit_noun}s, the plot counts "
            f"{self._member_noun}."
        )
        self._more.hide()
        self._finish_plots()

    def _finish_plots(self) -> None:
        """Park the slack under the rows, and say what the marks mean, once.

        A row keeps the height it was given: without the stretch the splitter shares its pane out
        among the wrappers, and three rows set to 114px drew at twice that. The package's
        top-justify rule, one container in.
        """
        self._plots_box.addStretch(1)
        self._legend.setText(self._axis_note())

    def _add_row(
        self, name: str, ranked, *, height: int | None = None, bare: bool = False
    ) -> DescriptorStrip:
        descriptor = DESCRIPTORS[name]
        strip = DescriptorStrip(descriptor.label, descriptor.unit)
        strip.set_row_height(height or self._row_px)
        strip.show_values(self._values[name], self._member_values.get(name, []))
        strip.picked.connect(lambda index, n=name: self._on_picked(n, index))
        if bare:
            # No name column in the drill-in: the heading already names the measurement, and the
            # column would indent this plot past the cluster lane stacked above it. Two plots
            # share an x axis only if their plotting areas start at the same x — otherwise the
            # tick marking a group's median stops sitting over its own data, which is the whole
            # reason the lane is drawn there.
            self._plots_box.addWidget(strip)
            self._rows.append((name, strip))
            return strip
        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        # A name and a number, and no pointing hand: the row above in the table is what opens
        # this measurement, and a cue on a label that opens nothing is a claim the wiring does
        # not back.
        side.addWidget(_label(descriptor.label, bold=True))
        if ranked is not None:
            side.addWidget(_label(fmt_value(ranked.value, descriptor.unit), muted=True))
        side.addStretch()
        holder = QWidget()
        holder.setLayout(side)
        holder.setFixedWidth(_name_column_px())
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(holder)
        row.addWidget(strip, 1)
        wrapper = QWidget()
        wrapper.setLayout(row)
        wrapper.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._plots_box.addWidget(wrapper)
        self._rows.append((name, strip))
        return strip

    def _show_table(self, chosen) -> None:
        """Every measurement, ranked, as a table whose rows open one.

        All of them, not just the drawn ones: "and it is ordinary on everything else" is the
        other half of a description, and a table listing only the shortlist cannot say it. The
        ``▸`` marks which already have a plot below, so the table says where you are as well as
        where you can go.
        """
        from detail.gui.pages import _table

        while self._table_box.count():
            widget = self._table_box.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._table = None
        if not self._ranked or self._open is not None:
            return

        drawn = {ranked.name for ranked in chosen}
        rows = []
        for ranked in self._ranked:
            descriptor = DESCRIPTORS[ranked.name]
            percent = ranked.percentile if ranked.above else 100.0 - ranked.percentile
            rows.append(
                (
                    descriptor.label,
                    fmt_value(ranked.value, descriptor.unit),
                    f"{'above' if ranked.above else 'below'} {percent:.0f}%",
                    f"{ranked.delta:+.2f}",
                    "strong" if ranked.strong else "",
                    "▸" if ranked.name in drawn else "",
                )
            )
        # ``ids``, not ``self._ranked[index.row()]``: ``_table`` turns sorting on, so a row's
        # position is only its rank until the first header click.
        self._table = _table(
            TABLE_HEADERS, rows, navigable=True, ids=[r.name for r in self._ranked]
        )
        # `_table` gives the slack to its last data column, which here is a one-glyph marker —
        # it filled half the width with nothing. Hand it to the name, the column that actually
        # varies in length; every column stays draggable either way.
        give_slack_to(self._table, 0)
        self._table.doubleClicked.connect(self._on_table_opened)
        self._table_box.addWidget(self._table)

    def _on_table_opened(self, index) -> None:
        name = row_id(index)
        if name is not None:
            self._open_measurement(name)

    def _scan_caption(self, drawn: int, strong: int) -> str:
        if not self._ranked:
            return ""
        if self._drawn_all:
            return f"Drawing all {len(self._ranked)}."
        return (
            f"Drawn: the top {drawn} of the {strong} it differs on."
            if strong
            else f"Drawn: the {drawn} largest effects, none of them strong."
        )

    def _axis_note(self) -> str:
        dense = any(row.is_dense for _n, row in self._rows)
        marks = (
            f"taller means more {self._member_noun}"
            if dense
            else f"each mark is one, and clicking it plays that {self._member_noun[:-1]}"
        )
        grey = (
            f"spans sampled across the recording"
            if self._baseline == BASELINE_RECORDING
            else f"all {self._member_noun} in the run"
        )
        note = (
            f"Grey is {grey}, colour is this {self._unit_noun}'s — {marks}. Each axis covers the "
            "middle 98% of its own measurement, so a value past either end sits on that end."
        )
        if self.posture_source:
            note += f" Postures come from {self.posture_source}."
        return note

    def _group_medians(self, name: str) -> list[float]:
        values = np.asarray(self._values[name], dtype=float)
        out = []
        for group in sorted(set(self._labels.tolist())):
            if str(group) == str(self._selected):
                continue
            rows = (self._labels == group) & np.isfinite(values)
            if rows.any():
                out.append(float(np.median(values[rows])))
        return out

    def _own_median(self, name: str) -> float:
        mine = _finite(self._member_values.get(name, []))
        return float(np.median(mine)) if len(mine) else float("nan")

    def _clear_rows(self) -> None:
        self._rows = []
        while self._plots_box.count():
            # `takeAt` returns spacers as well as widgets, and the trailing stretch is a spacer:
            # dropping only the widgets leaves one behind on every redraw.
            item = self._plots_box.takeAt(0)
            discard(item.widget())

    def _open_measurement(self, name: str | None) -> None:
        self._open = name
        self._drawn_all = False
        self._render()

    def _toggle_all(self) -> None:
        self._drawn_all = not self._drawn_all
        self._render()

    def _on_picked(self, name: str, index: int) -> None:
        """Play the member behind the mark, and say which one it was.

        The same answer to the same gesture as the map's squares: a point you can see is odd is
        only useful if you can then look at it.
        """
        members = self._members.get(name) or []
        if not (0 <= index < len(members)):
            return
        instance = members[index]
        player = self._ensure_player()
        player.set_clip(
            ClipSpec(
                recording_id=instance.recording_id,
                track_id=instance.subject if instance.subject is not None else 0,
                embed_run_id=self._embed_run_ids.get(instance.recording_id, ""),
                start_t_ns=int(instance.t_ns),
                end_t_ns=int(instance.end_t_ns),
                label=f"{instance.recording_id} {instance.t_ns / 1e9:.2f}s",
            )
        )
        seconds = (int(instance.end_t_ns) - int(instance.t_ns)) / 1e9
        self._played.setText(
            f"{DESCRIPTORS[name].label} · {instance.recording_id} at "
            f"{instance.t_ns / 1e9:.1f}s · {seconds:.2f}s long"
        )
        self._played.show()
        self.instance_picked.emit(instance)

    def _ensure_player(self) -> _ClipPlayer:
        """The player, made the first time there is something to play.

        Not before: an empty pane held a third of the width from the moment the tab opened, which
        is why the plots were too thin to read — ``gui_design``'s rule that a surface with nothing
        in it yet shows the one action, not the empty frame.
        """
        if self._player is None:
            self._player = _ClipPlayer(self._experiment, autoplay=True, context_s=0)
            self._split.addWidget(self._player)
            self._split.setStretchFactor(0, 2)
            self._split.setStretchFactor(1, 1)
            # Sized as well as weighted: a stretch factor only distributes space *past* each
            # pane's hint, and the rows column asks for little.
            if not self._split.remember():
                self._split.setSizes([2000, 1000])
        return self._player

    # -- measuring: on a button, off the UI thread, checked before every draw ------------- #
    def _group_name(self, group):
        """``group`` as a person names it: the selected one by the name its caller gave."""
        if group is not None and str(group) == self._selected:
            return self._selected_name
        return group

    def select(self, run, source_label, name: str | None = None) -> None:
        """Draw the members carrying ``source_label``, if the run is measured and still states
        what was measured. ``name`` is how the band names that group — a behavior's name, where
        ``source_label`` is its id — and defaults to the id.

        The **id**, not a predicate that selects on it. The ranking needs to know which group is
        open, so a caller passing a closure would have to pass the id as well — two ways of
        saying one thing, which is how a mask and a ranking come to disagree about what is
        selected while both render perfectly.

        Nothing slow starts here. An unmeasured run shows the button; a measured one is checked
        against the run and partitioned off-thread — a read of the artifact and a filter over rows
        already measured.
        """
        self._run = run
        self._selected = None if source_label is None else str(source_label)
        # The id is the key — it survives a rename — and never the label: "Behavior q01".
        self._selected_name = self._selected if name is None else str(name)
        self._generation += 1
        if self._measuring:
            return  # the measure in flight checks and draws the selection when it lands
        if self._measured is None or self._run_id != getattr(run, "id", None):
            self._show_pending(
                f"Measuring reads the track behind every one of this run's {self._member_noun}, "
                "which can take a minute or more on a large run, so it waits until you ask.",
                MEASURE,
            )
            return
        generation, measured, selected = (
            self._generation,
            self._measured,
            self._selected,
        )
        run_busy(
            self,
            lambda: _members_if_current(measured, run, self._experiment, selected),
            lambda payload: self._deliver(generation, payload),
            label="Checking the measurements…",
        )

    def recheck(self) -> None:
        """The run was edited: check the measurements against it before drawing them again.

        Each embedder connects the edit signal its other tabs already follow. Waiting for the next
        click is not enough, because an edit does not always move the selection — a committed
        cutoff leaves the same behavior selected — and the band would go on showing the old cut.
        """
        if self._run is not None:
            self.select(self._run, self._selected, self._selected_name)

    def measure_now(self) -> None:
        """Measure the run, with a bar. The button's slot, and the only place measuring starts."""
        run = self._run
        if run is None or self._measuring:
            return
        self._measuring = True

        def work(emit):
            # The marker stays on the *opening* line: the scan in
            # `tests/gui/test_ui_thread_blocking.py` matches per line, and black moves a trailing
            # comment to the closing paren when it wraps the call — which has silently dropped
            # this guard twice now.
            return measure_run(  # off-thread
                run, self._experiment, baseline=self._baseline, progress=emit
            )

        run_progress(
            self,
            work,
            lambda measured: self._on_measured(run, measured),
            label="Measuring…",
            on_error=self._on_measure_failed,
        )

    def _on_measured(self, run, measured: "Measured") -> None:
        self._measuring = False
        self._measured = measured
        self._run_id = getattr(run, "id", None)
        self._embed_run_ids = dict(measured.embed_run_ids)
        self.posture_source = ", ".join(sorted(measured.posture_runs)) or None
        # Through the check rather than straight to a draw, so an edit made while the bar ran
        # shows as one instead of being drawn under the new ids.
        self.select(self._run, self._selected, self._selected_name)

    def _on_measure_failed(self, exc: BaseException) -> None:
        self._measuring = False
        self.recheck()
        QMessageBox.critical(self, "Could not measure", str(exc) or type(exc).__name__)

    def _deliver(self, generation: int, payload) -> None:
        if generation != self._generation:
            return  # a later selection or edit has already asked again
        changed, partitioned = payload
        if changed is None:
            self.show_measurements(*partitioned)
            return
        self._show_pending(
            (
                f"This run has changed since it was measured, so the measurements no longer "
                f"match its {self._unit_noun}s."
                if changed == RUN_CHANGED
                else f"The pose run {changed}, which the postures come from, has changed "
                "since this run was measured, so the measurements no longer match it."
            ),
            REMEASURE,
        )

    def _show_pending(self, note: str, button: str) -> None:
        """Nothing drawn: one sentence saying why, and the button that changes it.

        Cleared, not dimmed. The previous numbers beside an "out of date" note is the state #515
        reported with a note added, and "these no longer describe it" is only true when none of
        them is on screen. The rows stay measured underneath, so an edit undone draws at once.
        """
        self._values, self._members, self._member_values = {}, {}, {}
        self._labels = np.asarray([], dtype=object)
        self._ranked, self._run_ranked, self._withheld = [], [], None
        self._open = None
        self._drawn_all = False
        self._clear_rows()
        self._show_table([])
        self._run_line.setText("")
        name = "" if self._selected is None else f" {self._selected_name}"
        self._heading.setText(f"{self._unit_noun.capitalize()}{name}")
        self._caption.setText("")
        self._legend.setText("")
        self._more.hide()
        self._back.hide()
        self._note.setText(note)
        self._note.show()
        self._measure_button.setText(button)
        self._measure_button.show()


#: What :func:`what_changed` returns when the run's own artifact moved, rather than a pose run's.
RUN_CHANGED = "run"


class Measured(NamedTuple):
    """What :func:`measure_run` measured, and what it was measured **from**.

    ``rows`` is ``{descriptor: [(label, value, instance), ...]}``. ``spans`` is the run's artifact
    as :func:`artifact_spans` states it, and ``posture_runs`` the same for each pose run whose
    postures were joined on, by id: every artifact the rows depend on, so a later draw can ask
    whether any of them has moved.
    """

    rows: dict
    spans: tuple
    posture_runs: dict
    embed_run_ids: dict


def artifact_spans(artifact) -> tuple:
    """Every instance's span and group, in the artifact's order — what a measurement rests on.

    The span decides a bout's numbers and the group decides which cluster they are drawn under,
    so two artifacts agreeing on this draw the same band. Either half alone misses an edit: the
    ids of clusters present at both cuts survive a granularity change, and every span survives a
    name that fuses nothing.
    """
    return tuple(
        (
            i.recording_id,
            i.subject if i.subject is not None else 0,
            int(i.t_ns),
            int(i.end_t_ns),
            str(i.source_label),
        )
        for i in artifact.instances
    )


def what_changed(measured: Measured, run, experiment) -> str | None:
    """Which artifact behind ``measured`` has moved since — :data:`RUN_CHANGED`, or a pose run's
    id — or ``None`` when every one still states what was measured.

    Off the UI thread: it reads the run's artifact and each posture source's. On large
    runs that read takes milliseconds, which is why it can run before every draw.
    """
    if artifact_spans(run.artifact()) != measured.spans:  # off-thread
        return RUN_CHANGED
    for run_id, spans in measured.posture_runs.items():
        source = experiment.authored_run(run_id)
        if source is None or artifact_spans(source.artifact()) != spans:  # off-thread
            return run_id
    return None


def _members_if_current(measured: Measured, run, experiment, selected):
    """``(changed, partitioned)``: what moved since ``measured``, or the band for ``selected``."""
    changed = what_changed(measured, run, experiment)
    if changed is not None:
        return changed, None
    return None, partition(measured.rows, selected)


def partition(rows: dict, selected) -> tuple:
    """``(run_wide, members, ids, labels)``: every measured value, and the ones in ``selected``."""
    run_wide, members, ids = {}, {}, {}
    labels: list = []
    for name, measured in rows.items():
        run_wide[name] = [value for _label, value, _ref in measured]
        if not labels:
            # `None` stays `None`: a background row belongs to no group, and stringifying it
            # would make it one — a "None" cluster in the run line and in the partition.
            labels = [
                None if label is None else str(label) for label, _v, _r in measured
            ]
        chosen = [
            (value, ref)
            for label, value, ref in measured
            if str(label) == str(selected)
        ]
        members[name] = [value for value, _ref in chosen]
        ids[name] = [ref for _value, ref in chosen]
    return run_wide, members, ids, labels


def measure_run(run, experiment, *, baseline: str, progress=None) -> Measured:
    """Every descriptor over everything worth measuring in a run, and what it was read from.

    Each row is ``(label, value, instance)``. ``label`` is the producer's own group id, or
    **``None`` for a background row** — a span that is measured and compared against but belongs
    to no group.

    ``baseline`` says what "the rest of the run" means here, and has no default because it decides
    whether the answer is right:

    * ``BASELINE_INSTANCES`` — the run's other instances. Right where the instances tile the
      recording, which is what a clustering run's bouts do: every segmented moment is in some
      cluster, so the other clusters *are* the rest.
    * ``BASELINE_RECORDING`` — spans sampled across each recording's whole trajectory. Right where
      the instances are a *selection* from the recording, which is what a query's matches are.
      Without it a one-behavior query run has nothing to compare against and the band draws
      nothing at all — four of the five real query runs in the reference experiment.

    ``progress(stage, fraction, detail)`` hears ``"writing frame views"`` first when a pose run the
    postures come from has views an edit deferred, then ``"measuring"``: weighted by the spans each
    unit will measure, folded within a unit, with an ETA in the detail.

    Off the UI thread: O(instances + background), and it reads a trajectory per subject.
    """
    from detail.analysis.descriptors import available, describe
    from detail.api.frame_slices import between
    from detail.core.progress import (
        ThrottledProgress,
        format_duration,
        remaining_seconds,
    )

    if baseline not in BASELINES:
        raise ValueError(
            f"unknown baseline {baseline!r}; expected one of {sorted(BASELINES)}"
        )

    artifact = run.artifact()  # off-thread
    spans_read = artifact_spans(artifact)
    by_unit: dict = {}
    for instance in artifact.instances:
        subject = instance.subject if instance.subject is not None else 0
        by_unit.setdefault((instance.recording_id, subject), []).append(instance)

    lengths = _span_lengths_ns(artifact.instances)
    # Each recording's own run: a run's recordings share embedding settings, not an id.
    embed_run_ids = run_selection.runs_by_recording(
        run.state(), {key[0] for key in by_unit}
    )
    candidates, posture_spans = _posture_candidates(experiment, by_unit, progress)

    # A unit costs the spans it will measure. Counting units would call a two-bout recording
    # half the work beside a two-thousand-bout one.
    background = BASELINE_WINDOWS if baseline == BASELINE_RECORDING else 0
    weights = {key: len(items) + background for key, items in by_unit.items()}
    total = max(sum(weights.values()), 1)
    throttled = ThrottledProgress(
        None if progress is None else (lambda event: progress(*event))
    )
    started = time.monotonic()

    def tick(done: float, index: int, recording_id: str, *, force=False) -> None:
        fraction = min(done / total, 1.0)
        detail = f"{recording_id} · {index + 1} of {len(by_unit)}"
        eta = remaining_seconds(fraction, time.monotonic() - started)
        if eta is not None:
            detail += f" · ~{format_duration(eta)} left"
        throttled(("measuring", fraction, detail), force=force)

    out: dict = {}
    names: list[str] = []
    used: set = set()
    done = 0.0
    for index, (key, unit_instances) in enumerate(by_unit.items()):
        tick(done, index, key[0], force=True)
        weight = weights[key]
        trajectory, source = _load_slice(
            experiment, key, embed_run_ids.get(key[0]), candidates.get(key, [])
        )
        if trajectory is None or not len(trajectory):
            done += weight
            continue
        if source is not None:
            used.add(source)
        spans = [(int(i.t_ns), int(i.end_t_ns), i) for i in unit_instances]
        if baseline == BASELINE_RECORDING:
            spans += [
                (start, end, None)
                for start, end in _baseline_spans(trajectory, lengths)
            ]
        step = weight / max(len(spans), 1)
        for position, (start, end, instance) in enumerate(spans, start=1):
            bout = between(trajectory, start, end)
            if not names:
                names = available(bout)
                if not names:
                    return Measured({}, spans_read, {}, embed_run_ids)
            label = None if instance is None else instance.source_label
            for name, value in describe(bout, names=names).items():
                out.setdefault(name, []).append((label, value, instance))
            tick(done + position * step, index, key[0])
        done += weight
    if by_unit:
        tick(total, len(by_unit) - 1, next(reversed(by_unit))[0], force=True)
    if baseline == BASELINE_RECORDING:
        # A background window is as long as we made it, so a descriptor of the *span* compares a
        # measurement against a parameter — `duration` read "above 100% of them", a fact about
        # the sampler wearing the clothes of a finding. Declared on the descriptor rather than
        # named here, so a second one added later is dropped without editing this.
        for name in [n for n in out if DESCRIPTORS[n].measures_span]:
            del out[name]
    return Measured(
        out,
        spans_read,
        {run_id: posture_spans[run_id] for run_id in used},
        embed_run_ids,
    )


def _span_lengths_ns(instances) -> np.ndarray:
    """The lengths a background window may take: the instances' own, to be resampled.

    Not their median. A baseline measured over a different span length is not a baseline — speed
    over 0.2s and speed over 20s are different quantities — but a baseline of *one* length is
    worse: every window is then exactly as long as every other, so a length-sensitive measurement
    becomes a fact about the sampler. It showed as ``duration 1.07 s · above 100%``, which reads
    as a finding and is a restatement of the window size.
    """
    spans = np.array(
        [int(i.end_t_ns) - int(i.t_ns) for i in instances if i.end_t_ns > i.t_ns],
        dtype=np.int64,
    )
    return spans if len(spans) else np.array([_NS_PER_S], dtype=np.int64)


def _baseline_spans(trajectory, lengths: np.ndarray) -> list[tuple[int, int]]:
    """Up to :data:`BASELINE_WINDOWS` spans spread evenly across a whole trajectory.

    Evenly across, not the first N: a run's opening minutes are not a sample of it. Each window
    takes a length drawn from ``lengths`` — the instances' own durations — so the pool matches
    what it is compared against in everything except *where* it sits. Capped because the pool only
    has to describe a distribution, and a few hundred spans fill a
    :data:`~detail.gui.plots.BINS`-bin histogram.

    Seeded, so re-opening a run twice draws the same picture. An unseeded baseline would move the
    grey spread under an unchanged selection with nothing on screen to say why.
    """
    times = np.asarray(trajectory.t_ns, dtype=np.int64)
    if len(times) < 2:
        return []
    first, last = int(times[0]), int(times[-1])
    drawn = np.random.default_rng(0).choice(lengths, size=BASELINE_WINDOWS)
    spans = []
    for start, length in zip(np.linspace(first, last, BASELINE_WINDOWS), drawn):
        end = min(int(start) + int(length), last)
        if end > int(start):
            spans.append((int(start), end))
    return spans


def _posture_candidates(experiment, units, progress) -> tuple[dict, dict]:
    """The pose runs that could lend each unit its postures, brought current, and what each
    states.

    ``({unit: [pose_run_id, ...]}, {pose_run_id: artifact_spans})``, newest first per unit.
    Resolved once, before anything is measured. It used to scan the experiment's runs once per
    unit, and a pose run named since its views were last read rebuilt them inside the first
    unit's read, where the bar could not say what it was doing (§21 defers those views to their
    first reader, and this is one).

    The spans are read **after** the views are brought current, so they describe the views the
    postures will be joined from.
    """
    from detail.storage import frame_view as frame_view_store

    pose_runs = sorted(
        (
            ref.id
            for ref in experiment.analysis_runs  # off-thread
            if ref.module_id == "pose_clustering"
        ),
        reverse=True,
    )
    candidates = {
        (recording_id, subject): [
            run_id
            for run_id in pose_runs
            if frame_view_store.unit_path(
                experiment.analysis_run_dir(run_id), recording_id, subject
            ).is_dir()
        ]
        for recording_id, subject in units
    }
    states: dict = {}
    for run_id in sorted({r for ids in candidates.values() for r in ids}, reverse=True):
        pose_run = experiment.authored_run(run_id)
        if pose_run is None:
            continue
        pose_run.bring_frame_views_current(progress=progress)
        states[run_id] = artifact_spans(pose_run.artifact())  # off-thread
    candidates = {
        unit: [r for r in ids if r in states] for unit, ids in candidates.items()
    }
    return candidates, states


def _load_slice(experiment, key, embed_run_id, pose_runs):
    """One track's whole trajectory with postures joined on, and the pose run they came from.

    ``(slice, pose_run_id)``: ``(None, None)`` when the track is unreadable, and a ``None`` pose
    run when no candidate labels this unit. Unreadable is ordinary — a tracking run deleted after
    the fit — and a reason to show fewer numbers rather than to fail opening the tab.
    """
    from detail.api.frame_slices import postures_from, track_slice

    recording_id, subject = key
    try:
        recording = experiment.get_recording(recording_id)
        record = next(
            (e for e in recording.embedding_runs if e.get("id") == embed_run_id), {}
        )
        # The sidecar when the run wrote one (mask centroid and area), the boxes otherwise:
        # the same rows the cluster profile and the clip window measure.
        slice_ = track_slice(recording, {"id": embed_run_id, **record}, subject)
        if slice_ is None:
            return None, None
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return None, None

    # Postures, when a pose run covers this unit. **Which** pose run, where there are several, is
    # chosen rather than refused — the most recent — because refusing means an empty panel; and
    # it is *stated*, in the caption, so the choice is a fact the reader can check rather than an
    # order-dependent default nobody can see.
    for run_id in pose_runs:
        joined = postures_from(
            experiment.analysis_run_dir(run_id),
            slice_,
            recording_id=recording_id,
            subject=subject,
        )
        if joined.labels is not None:
            return joined, run_id
    return slice_, None
