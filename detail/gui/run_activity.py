"""The live-run panel: a bar, a status line, and a row per part of the run.

**Why this is one object and not a thing each page builds.** Two pages drove a live run and each
built the same scaffold — `RunDetailPage` and `_FitDetailPage`, the latter serving five detail
pages of its own. Line for line the same: a `QProgressBar`, a `QLabel`, a `tick`/`finished_run`
pair, and a `_start_live` that rebinds the worker on a retry. They differed in exactly two ways —
which worker class, and whether `finished_run` carries an error string.

Adding anything to both copies is this package's recorded *"two surfaces, two mechanisms"* class:
the next capability lands on whichever page its author had open, and the two drift until they
disagree about what a run looks like. So the scaffold lives here and both pages embed it.

**This held an Activity band — a timestamped log of every progress update — and no longer does.**
It was added because the page had one bar and one replaced label, so nothing on it could say that
nothing had happened for two hours. The unit rows answer that better and per part, which left the
band restating them one line of text at a time: *"22:12:40 · recording_A (1/2) · 1% · ~1h57m
left"* above a row already reading `detecting · 1% · ~1h57m left`.

`activity.csv` is still written for every run (`storage/activity.py`). It is what the performance
export reads and what a bug report attaches; what went away is a widget restating the rows above
it. What is genuinely lost with it is the *history* of a finished run inside the app — the file
still has it, no screen shows it.
"""

from __future__ import annotations

from pathlib import Path

from detail.core.progress import format_duration
from detail.gui.collapse import CollapsibleBand
from detail.gui.qt import (
    QLabel,
    QListWidget,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    Qt,
)
from detail.gui.sizing import fit_list_height, lines, mono_font
from detail.gui.unit_rows import UnitRows
from detail.storage import activity

#: How many rows the band holds. A ten-hour run writes thousands; a person reads the last screen
#: of them, and the rest are in the file for a bug report. Keeping them all in a list widget is
#: the cost of a scrollback nobody scrolls.
VISIBLE_ROWS = 200

#: How many rows the band is tall before it starts scrolling its own content.
#:
#: The list twin of `tables.MAX_VISIBLE_ROWS`, and the same rule: a `QListWidget` is `Expanding`,
#: so left alone a three-line log opens a 200px void — the "325px of dead grid" complaint, one
#: widget over. Larger than `fit_list_height`'s default of 8 because this one is read as a
#: *history*: the question it answers is whether anything has happened lately, which takes more
#: than a couple of lines to see.
BAND_ROWS = 14


class LiveRunPanel(QWidget):
    """Bar + status + a row per part of the run, for a page that drives a live one.

    **Every widget in here has a pinned height, and that is the invariant.** This panel updates
    roughly twice a second for hours, and it sits at the top of a page of ordinary content — so
    any height in it that is *derived* from text becomes the whole page below twitching. The
    status line learned this first (it wrapped, so a status growing a second line shoved
    everything down); the rule is now the panel's rather than that one widget's.

    It is only half the job. The other half is on the page: a trailing `addStretch(1)`, so the
    slack lands in the spacer instead of being shared out among the form rows above it. With a
    0-factor spacer the form spreads to fill — measured 60px between single-line rows against 21
    compact — and a spread form is a form whose rows *redistribute* whenever anything above it
    changes by a pixel.
    """

    def __init__(self) -> None:
        super().__init__()
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar()
        # Pinned, like everything else in this panel. A progress bar's height is a style metric
        # and not obviously variable — but this panel sits above a page of content, and the whole
        # point of the panel is that it updates about twice a second. Anything in it whose height
        # is *derived* is a thing that can move the page under someone's cursor, so nothing in it
        # is derived.
        self.progress.setFixedHeight(lines(1.4))
        box.addWidget(self.progress)
        self.status = QLabel("")
        # One line, always, and it never pushes on the layout either way.
        #
        # It used to wrap, because a non-wrapping label reports its full text width as
        # `minimumSizeHint` and that inflated the right pane's minimum — locking the main
        # splitter while a live run streamed a long status. But a wrapping label changes
        # *height* as the text changes, which moves everything under it on a page that updates
        # twice a second. `Ignored` solves the original problem properly (fill what you are
        # given, never push back), and a fixed height of one line solves the new one.
        self.status.setWordWrap(False)
        self.status.setFixedHeight(lines(1))
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        box.addWidget(self.status)
        # The run's parts, above the log. This is the page's primary progress display: the bar
        # says how far the *run* is, and these say what is actually happening — which was the
        # thing a person had to read a scrolling log to work out.
        # A heading, because these rows say *what the run is made of* and that differs by run
        # kind — recordings, phases, stages. An unlabelled grid under a bar reads as more of the
        # bar; the surface names its own units.
        self.units_heading = QLabel("")
        heading_font = self.units_heading.font()
        heading_font.setBold(True)
        self.units_heading.setFont(heading_font)
        self.units_heading.setVisible(False)
        box.addWidget(self.units_heading)

        self.units = UnitRows()
        self.units.setVisible(False)
        box.addWidget(self.units)

        self._active = False

    def set_live(self, live: bool) -> None:
        """Show the bar and the status while this page is streaming a run.

        `live` means *this page is driving the bar*, which needs a worker behind it or it sits
        frozen at zero — worse than no bar at all. Whether the run is **running** is a different
        fact and comes from the record: see `set_active`.
        """
        self.progress.setVisible(live)
        self.status.setVisible(live)

    def set_active(self, active: bool) -> None:
        """Whether the run is going, as the *record* tells it rather than the page handle.

        It no longer forces the band open, and that is a consequence of the unit rows existing.
        The band was opened on a running run because it was the only thing on the page that said
        what was happening; the rows are that now, and better — so the fold goes back to being
        the user's, remembered, and the band answers the narrower question it is actually good
        at: whether anything has happened at all in the last two hours.

        Kept as a method rather than deleted because the pages call it with the fact, and what a
        panel does with "this run is going" is the panel's business to change.
        """
        self._active = active

    def tick(self, permille: int, status: str) -> None:
        self.progress.setMaximum(1000)
        self.progress.setValue(permille)
        self.status.setText(status)

    def set_units(self, units, *, heading: str = "") -> None:
        """Show the run's parts, or hide the grid when a run has none to show.

        Hidden rather than emptied: an empty grid is a claim that the run has no parts, and a
        run with nothing to break into genuinely has none — the overall bar already says
        everything there is to say about it.
        """
        self.units.setVisible(bool(units))
        self.units_heading.setText(heading)
        self.units_heading.setVisible(bool(units and heading))
        if units:
            self.units.set_units(units)

    def _units_drawn(self) -> list:
        """The units currently on screen — empty when the grid is hidden.

        An accessor rather than reaching into the child widget from a test: what is *drawn* is
        the property these guards are about, and a hidden grid holding stale units is exactly
        the state that would otherwise read as a pass.
        """
        return list(self.units._units) if self.units.isVisibleTo(self) else []
